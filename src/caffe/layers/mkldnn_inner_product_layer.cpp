#ifdef MKLDNN_SUPPORTED
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkldnn_layers.hpp"
#include "mkl_service.h"

#if 0
#include "mkldnn_types.h"

using namespace mkldnn;
#endif

// TODO: Add transposed weights support

namespace caffe {
template <typename Dtype>
MKLDNNInnerProductLayer<Dtype>::MKLDNNInnerProductLayer(const LayerParameter& param)
            : MKLDNNLayer<Dtype>(), InnerProductLayer<Dtype>(param)
            , fwd_bottom_data(NULL), fwd_top_data(NULL), fwd_weights_data(NULL), fwd_bias_data(NULL)
            , ipFwd_pd(NULL), output_memory(NULL)
            , input_primitive(NULL), weights_primitive(NULL), bias_primitive(NULL)
            , w_(0), h_(0)
{
}

template <typename Dtype>
MKLDNNInnerProductLayer<Dtype>::~MKLDNNInnerProductLayer()
{
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom
                                            , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNInnerProductLayer<Dtype>::LayerSetUp: " << this->layer_param_.name();
    InnerProductLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom
                                            , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNInnerProductLayer<Dtype>::Reshape: " << this->layer_param_.name();
    InnerProductLayer<Dtype>::Reshape(bottom, top);

    this->w_ = bottom[0]->width();
    this->h_ = bottom[0]->height();
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::InitInnerProduct(const vector<Blob<Dtype>*>& bottom
                                                    , const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;

    int32_t n  = this->M_;
    int32_t w = this->w_;
    int32_t h = this->h_;
    int32_t oc = this->N_;
    int32_t ic = this->K_/h_/w_;
    bool has_spatial = h > 1 || w > 1;

    // Initialize memory descriptors (fromat = any) to create inner_product descriptor
    memory::precision mpcsn = memory::precision::f32;
    memory::format mfmt = memory::format::any;

    tensor::dims input_tz = (has_spatial) ? tensor::dims{n, ic, h, w} : tensor::dims{n, ic};
    tensor::dims output_tz = {n, oc};
    tensor::dims weights_tz = (has_spatial) ? tensor::dims {oc, ic, h, w} : tensor::dims{oc, ic};
    tensor::dims bias_tz = {oc};

    memory::desc init_input_md({input_tz}, mpcsn, mfmt);
    memory::desc init_output_md({ output_tz}, mpcsn, mfmt);
    memory::desc init_weights_md({weights_tz}, mpcsn, mfmt);
    memory::desc init_bias_md({bias_tz}, mpcsn, mfmt);

    // Initialize inner_product primitive descriptor
    inner_product::desc ipFwd_desc(prop_kind::forward, init_input_md, init_weights_md
                                                ,init_bias_md, init_output_md);

    engine cpu_engine = CpuEngine::Instance().get_engine();

    ipFwd_pd.reset(new inner_product::primitive_desc(ipFwd_desc, cpu_engine));

    // Create priv memory primitive descriptors stored as class members
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    shared_ptr<MemPD> prv_input_memory_pd(new MemPD(ipFwd_pd->data.src_primitive_desc));
    shared_ptr<MemPD> prv_bias_memory_pd(new MemPD(ipFwd_pd->data.bias_primitive_desc));
    shared_ptr<MemPD> prv_output_memory_pd(new MemPD(ipFwd_pd->data.dst_primitive_desc));
    shared_ptr<MemPD> prv_weights_memory_pd(new MemPD(ipFwd_pd->data.weights_primitive_desc));

    // Create usr memory primitive descriptors stored as class members
    memory::format input_mfmt = has_spatial ? memory::format::nchw : memory::format::nc;
    shared_ptr<MemPD> usr_input_memory_pd(new MemPD({{input_tz}, mpcsn, input_mfmt}, cpu_engine));
    shared_ptr<MemPD> usr_bias_memory_pd(new MemPD({{bias_tz}, mpcsn, memory::format::x}, cpu_engine));
    shared_ptr<MemPD> usr_output_memory_pd(new MemPD({{output_tz}, mpcsn, memory::format::nc}, cpu_engine));
    memory::format weights_mfmt = has_spatial ? memory::format::oihw : memory::format::oi;
    shared_ptr<MemPD> usr_weights_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_input_memory_pd, prv_input_memory_pd, bottom[0], this));
    input_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_output_memory_pd, prv_output_memory_pd, top[0], this));
    output_memory = fwd_top_data->create_output_memory();

    fwd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_memory_pd, prv_weights_memory_pd, this->blobs_[0].get(), this));
    weights_primitive = fwd_weights_data->create_input(false);

    if (this->bias_term_) {
        fwd_bias_data.reset(new MKLDNNData<Dtype>(usr_bias_memory_pd, prv_bias_memory_pd, this->blobs_[1].get(), this));
        bias_primitive = fwd_bias_data->create_input(false);
    }
    ipFwd.reset(new inner_product(prop_kind::forward
                            , *input_primitive, *weights_primitive
                            , *bias_primitive, *output_memory));
    fwd_bottom_data->set_mkldnn_primitive(ipFwd);
    fwd_top_data->set_mkldnn_primitive(ipFwd);
    fwd_weights_data->set_mkldnn_primitive(ipFwd);
    fwd_bias_data->set_mkldnn_primitive(ipFwd);

    // Names are for debugging purposes only.
    fwd_bottom_data ->name = "fwd_bottom_data   @ " + this->layer_param_.name();
    fwd_top_data    ->name = "fwd_top_data      @ " + this->layer_param_.name();
    fwd_weights_data->name = "fwd_weights_data  @ " + this->layer_param_.name();
    fwd_bias_data   ->name = "fwd_bias_data     @ " + this->layer_param_.name();
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNInnerProductLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
    if( ipFwd_pd == NULL)
        InitInnerProduct(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read(false);
    fwd_weights_data->sync_before_read(true);
    fwd_bias_data->sync_before_read(true);
    // update top that head at prv
    fwd_top_data->sync_before_write();

    ipFwd.submit();
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                                , const vector<bool>& propagate_down
                                                , const vector<Blob<Dtype>*>& bottom)
{
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MKLDNNInnerProductLayer);
#else

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top
                                                , const vector<bool>& propagate_down
                                                , const vector<Blob<Dtype>*>& bottom)
{
    NOT_IMPLEMENTED;
}
#endif

INSTANTIATE_CLASS(MKLDNNInnerProductLayer);
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
