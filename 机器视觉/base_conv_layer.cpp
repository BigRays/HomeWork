#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

//// ����˵���״[kernel_h, kernel_w]
// Blob<int> kernel_shape_;

//// ������״[stride_h, stride_w]
// Blob<int> stride_;

//// padding��״[pad_h, pad_w]
// Blob<int> pad_;

//// ���ž������״�������ο�ʽ�ľ��
// Blob<int> dilation_;

//// �����������״ = [����ͼ��ͨ����, ����ͼ��h, ����ͼ��w]
// Blob<int> conv_input_shape_;

//// col_buffer����״ = [kernel_dim_, conv_out_spatial_dim_ ]
//// ����ͼ��ת�������ھ����չ����col��ʽ������ο�src/utils/im2col.cpp��������col_buffer���������Ȩֵ��col_buffer=����������������״Ϊ�������ӡ�
// vector<int> col_buffer_shape_;

//// ���������״������vector��
// vector<int> output_shape_;

//// ���������״������vector�����ָ�룬��Ϊ�Ǳ�Ĳ�������ֱ����ָ��ָ��֮ǰ�Ѿ����ڵ���һ���output_shape_��
// const vector<int>* bottom_shape_;

//// �ռ�����������������Ǽ�άͼ��
// int num_spatial_axes_;

//// �����ά�� = ����ͨ����*����ͼ���h*����ͼ���w
// int bottom_dim_;

//// ���ά�� = ���ͨ����*���ͼ���h*���ͼ���w
// int top_dim_;

//// ����ͼ����ĸ�axis��channel,һ���ǵڶ���ά��
// int channel_axis_;

//// �Ѵ�С
// int num_;

//// ͨ����
// int channels_;

//// �����Ĵ�С
// int group_;

//// ����ռ�ά�� = ���֮���ͼ��*���֮��ͼ��Ŀ�
// int out_spatial_dim_;

//// ʹ�þ�����õ���Ȩֵƫ��
// int weight_offset_;

//// ������ͼ���ͨ����
// int num_output_;

//// �Ƿ�����ƫ��
// bool bias_term_;

//// �Ƿ���1x1���
// bool is_1x1_;

//// ǿ��ʹ��nάͨ�þ������im2col��nά��ʽ�������Ǹ����õĶ�ά��ʽ��
// bool force_nd_im2col_;

// im2col��ͼ�����ľ������ͨ��im2colת��Ϊ��Ч�ľ���˷����㡣
// ��������ͼ��ľ�������������˲��������������ɨ��ͼ��Ĺ��̣���ÿ�������ڶ����˲���Ȩֵ��ͼ�������ֵ֮��ļ򵥵�����
// ��ˣ����԰�ÿ��ͼ���չ��һ�У����˲���չ��һ�У����о���˷���ʵ�־������

// CHECK_EQ(x, y) << "x!=y"��EQ��equation����Ϊ�����ڡ��������ж��Ƿ�x����y����x != yʱ��������ӡ��x != y��
//
// CHECK_NE(x, y) << "x=y"��NE��not equation����Ϊ�������ڡ��������ж��Ƿ�x������y����x = yʱ��������ӡ��x = y��
//
// CHECK_LE(x, y) << "x<=y", LE��lower equation, ��ΪС�ڵ��ڣ������ж��Ƿ�xС�ڵ���y����x <= yʱ��������ӡx <= y��
//
// CHECK_LT(x, y) << "x<y", LT��Ϊlower to ����ΪС�ڣ������ж��Ƿ�xС��y����x < yʱ��������ӡx < y��
//
// CHECK_GE(x, y) << "x>=y", GE��Ϊgreat equation����Ϊ���ڵ��ڣ������ж��Ƿ�x���ڵ���y����x >= yʱ��������ӡx >= y��
//
// CHECK_GT(x, y) << "x>y", GT��Ϊgreat to ����Ϊ���ڣ������ж��Ƿ�x����y����x > yʱ��������ӡx > y��


// base_conv_layer.cpp �ж����� BaseConvolutionLayer ���һЩ��Ա�������� BaseConvolutionLayer �� ConvolutionLayer �ĸ���
// ConvolutionLayer ���õ���һЩ�����������ﶨ�壬�����ڿ� conv_layer ǰ��Ҫ����Դ���롣
namespace caffe {
	// ���������Ҫ�Ǹ��� protobuf �еĲ�������ã����þ���˵Ĵ�С��padding�����������롢Ȩֵ������ʼ���ȵȡ��ǲ㽨���Ļ������Ǵ��㽨���ĵ�һ������������ü���ʼ����
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// Configure the kernel size, padding, stride, and inputs.
		// ����protobuf�еĲ�������ã����þ���˵Ĵ�С��padding������������ȵȡ�
		ConvolutionParameter conv_param = this->layer_param_.convolution_param();	// �������
		force_nd_im2col_ = conv_param.force_nd_im2col();	// ���ݲ���������Ƿ�ǿ�ƽ���nάim2col
		// ����ĺ���������blob.hpp�����ڼ������index�Ƿ��ڷ�Χ�ڣ���֧�����븺�����������������������������
		// ȷ���ĸ�ά����channel��һ���ǣ�batch*channel*height*width���ڶ���ά�ȡ�
		/* channel_axis_���������ȡ���������е�axis������Ĭ��Ϊ1����ʾ��channel��ͣ�����blobΪ(N,C,W,H)ʱ��
		һ�����ͨ����Ӧ�����о���˶�����blob�ϸ�ͨ������ά�������������ͨ������Ľ������������Ϊ
		һ�������������ͼ */
		channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
		// channel������ľ���ͼ��ߡ������ֿռ��ᣬ��һ���ռ����ǵڼ�ά��Ȼ����+1
		const int first_spatial_axis = channel_axis_ + 1;	// ָʾ�������ͼ��ĵ�һ���ᣬ������H(height)
		const int num_axes = bottom[0]->num_axes();	// �õ�bottom blob��ά��
		num_spatial_axes_ = num_axes - first_spatial_axis;	// �ռ����������������ά����
		CHECK_GE(num_spatial_axes_, 0);	// ��������ά�����������0

		vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);	// ���ڳ�ʼ����������������ݵ���״��һ����ά(C,H,W)
		// ��num_spatial_axes_==2ʱ��spatial_dim_blob_shape���vectorֻ����һ��Ԫ����ֵΪ2
		vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));	//���ڳ�ʼ������˵���״
		// �����˲��������ά��
		// Setup filter kernel dimensions (kernel_shape_).
		// ����blob.cpp��� void Blob<Dtype>::Reshape(const vector<int>& shape)
		// ��spatial_dim_blob_shapeΪ����������һ��Blob����kernel_shape_�������Blob��ά����Ϣֻ����һ��ά�ȣ�ֵΪ2,
		// Ҳ����˵���Blob��count_==2���������Blob��ά����Ϣֻ����һ��ά��,ֻ����������
		// ��Ϊ�ں����ļ��㣨Im2col���У���ֻ�������Blob�е����ݵ�ֵ�������������Blob��shape��Ϣ��
		// ������Im2col()�У�ֻҪȡ����Ӧ��ֵ����kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1]��
		kernel_shape_.Reshape(spatial_dim_blob_shape);	// ��ʼ������˵���״(��*��)
		int* kernel_shape_data = kernel_shape_.mutable_cpu_data();	// �õ���¼�������״���ݵ�ַ
		// ����������ж������˸߻���������kernel_shape_data��һblob��
		/* ����������û���Զ����ά����ľ���˳�������ж�����ֱ�ֵ�����Զ����˶�ά�����
		����Ļ���kernal_size���������ܱ����壬����Ƿ�����������û�ж����ά����˵ĳ�����ô����
		kernal_size����������˸�ֵ�������һ���������� */
		if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
			CHECK_EQ(num_spatial_axes_, 2)
				<< "kernel_h & kernel_w can only be used for 2D convolution.";
			CHECK_EQ(0, conv_param.kernel_size_size())
				<< "Either kernel_size or kernel_h/w should be specified; not both.";
			// ����˵ĸ�
			kernel_shape_data[0] = conv_param.kernel_h();
			// ����˵Ŀ�
			kernel_shape_data[1] = conv_param.kernel_w();
		}
		// ���������û�ж������˿�͸ߣ�����ݾ���˵�ά������ȷ������һά�˴�С�͸���
		else {
			const int num_kernel_dims = conv_param.kernel_size_size();
			CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
				<< "kernel_size must be specified once, or once per spatial dimension "
				<< "(kernel_size specified " << num_kernel_dims << " times; "
				<< num_spatial_axes_ << " spatial dims).";
			for (int i = 0; i < num_spatial_axes_; ++i) {
				kernel_shape_data[i] =
					conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
			}
		}
		// ������˲���(�߿�)�Ƿ�Ϸ�
		for (int i = 0; i < num_spatial_axes_; ++i) {
			CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
		}
		// ��������stride_��pad_��dilation_��Щblob���趨Ҳ���ơ�
		// ���ò���ά��
		// Setup stride dimensions (stride_)
		stride_.Reshape(spatial_dim_blob_shape);
		int* stride_data = stride_.mutable_cpu_data();
		/* ����������û���Զ����ά���ʱ�ߺͿ���Ĳ����������������ֵ�����û�ж���Ļ����Ͱ�������
		�������������ļ��еľ�����stride������ֵ��stride����Ҫ��ȱʧ�Ļ�����Ĭ��ΪkDefaultStride����Ϊ1��
		��������ֻ������һ������ֵ������ߺͿ���Ĳ���һ�¡� */
		if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
			CHECK_EQ(num_spatial_axes_, 2)
				<< "stride_h & stride_w can only be used for 2D convolution.";
			CHECK_EQ(0, conv_param.stride_size())
				<< "Either stride or stride_h/w should be specified; not both.";
			stride_data[0] = conv_param.stride_h();
			stride_data[1] = conv_param.stride_w();
		}
		else {
			const int num_stride_dims = conv_param.stride_size();
			CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
				num_stride_dims == num_spatial_axes_)
				<< "stride must be specified once, or once per spatial dimension "
				<< "(stride specified " << num_stride_dims << " times; "
				<< num_spatial_axes_ << " spatial dims).";
			const int kDefaultStride = 1;
			for (int i = 0; i < num_spatial_axes_; ++i) {
				stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
					conv_param.stride((num_stride_dims == 1) ? 0 : i);
				CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
			}
		}
		// Setup pad dimensions (pad_).
		/* ����������û���Զ���ߺͿ����pad�������������ֵ�����û�ж���Ļ����Ͱ�������
		�������������ļ��еľ�����pad������ֵ��pad����Ҫ��ȱʧ�Ļ�Ĭ��ΪkDefaultPad����Ϊ0��
		��������ֻ������һ��padֵ������ߺͿ����padһ�¡� */
		pad_.Reshape(spatial_dim_blob_shape);
		int* pad_data = pad_.mutable_cpu_data();
		if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
			CHECK_EQ(num_spatial_axes_, 2)
				<< "pad_h & pad_w can only be used for 2D convolution.";
			CHECK_EQ(0, conv_param.pad_size())
				<< "Either pad or pad_h/w should be specified; not both.";
			pad_data[0] = conv_param.pad_h();
			pad_data[1] = conv_param.pad_w();
		}
		else {
			const int num_pad_dims = conv_param.pad_size();
			CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
				num_pad_dims == num_spatial_axes_)
				<< "pad must be specified once, or once per spatial dimension "
				<< "(pad specified " << num_pad_dims << " times; "
				<< num_spatial_axes_ << " spatial dims).";
			const int kDefaultPad = 0;
			for (int i = 0; i < num_spatial_axes_; ++i) {
				pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
					conv_param.pad((num_pad_dims == 1) ? 0 : i);
			}
		}
		// Setup dilation dimensions (dilation_).
		/* ����������û���Զ���ߺͿ���ľ������չ�������������ֵ�����û�ж���Ļ����Ͱ�������
		�������������ļ��еľ�����dilation������ֵ��dilation_����Ҫ��ȱʧ�Ļ�Ĭ��ΪkDefaultDilation��
		��Ϊ1����ʾ����˲�������չ�� */
		dilation_.Reshape(spatial_dim_blob_shape);
		int* dilation_data = dilation_.mutable_cpu_data();
		const int num_dilation_dims = conv_param.dilation_size();
		CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
			num_dilation_dims == num_spatial_axes_)
			<< "dilation must be specified once, or once per spatial dimension "
			<< "(dilation specified " << num_dilation_dims << " times; "
			<< num_spatial_axes_ << " spatial dims).";
		const int kDefaultDilation = 1;
		for (int i = 0; i < num_spatial_axes_; ++i) {
			dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
				conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
		}
		// Special case: im2col is the identity for 1x1 convolution with stride 1
		// and no padding, so flag for skipping the buffer and transformation.
		// �ж��ǲ���1*1���
		is_1x1_ = true;
		for (int i = 0; i < num_spatial_axes_; ++i) {
			is_1x1_ &=
				kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
			if (!is_1x1_) { break; }
		}
		// Configure output channels and groups.
		channels_ = bottom[0]->shape(channel_axis_);	// ����channel��ά������������blob��channel������ȡ���������ĵ�blob��ͨ������
		num_output_ = this->layer_param_.convolution_param().num_output();	// ��ȡ����������ͨ����
		CHECK_GT(num_output_, 0);	// �������ͨ�����Ƿ������
		group_ = this->layer_param_.convolution_param().group();	// ��ȡ������С
		// ���롢��� channel ��������Ϊgroup����������ÿ��group�еľ����ֻ�Ա�group��ӦƵ��������ͼ���о������
		CHECK_EQ(channels_ % group_, 0);	// ��������ĵ�blobͨ�����Ƿ��ܱ������������
		CHECK_EQ(num_output_ % group_, 0)	// �������ͨ�����Ƿ��ܱ������������
			<< "Number of output should be multiples of group.";
		// ����Ҫ��ת����������򽻻�������������򲻽���
		if (reverse_dimensions()) {
			conv_out_channels_ = channels_;
			conv_in_channels_ = num_output_;
		}
		else {
			conv_out_channels_ = num_output_;
			conv_in_channels_ = channels_;
		}
		// Handle the parameters: weights and biases.
		// - blobs_[0] holds the filter weights
		// - blobs_[1] holds the biases (optional)
		// Ȩֵ��״����conv_out_channels_��conv_in_channels_ / group_��kernel_h, kernel_w��,�磺256*256*3*3
		vector<int> weight_shape(2);	// ��������������
		weight_shape[0] = conv_out_channels_;	// Ȩ�ز���shape�ĵ�һ����Ϊ���ͨ����С����ÿ�����ͨ����Ӧ���Եľ���ˣ����Ϊnum
		// ÿ��group�еľ����ֻ�Ա�group��ӦƵ��������ͼ���о����������������������3x3x32��3x3x16������group��Ӱ�죩
		weight_shape[1] = conv_in_channels_ / group_;	// Ȩ�ز���shape�ĵڶ�����Ϊ����ͨ����С���Ծ�����������Ϊchannel
		for (int i = 0; i < num_spatial_axes_; ++i) {
			weight_shape.push_back(kernel_shape_data[i]);	// Ȩ�ز���shape�ĵ������͵��ĸ���Ϊ�����ά�ȴ�С
		}
		bias_term_ = this->layer_param_.convolution_param().bias_term();	// ��ȡ�Ƿ�ʹ��ƫ�õĲ���
		vector<int> bias_shape(bias_term_, num_output_);	// ����ƫ�ò��������bias_term_Ϊtrue(1)����ôbias_shape[0]=num_output_
		if (this->blobs_.size() > 0) {
			CHECK_EQ(1 + bias_term_, this->blobs_.size())	// ����blobs_�Ƿ�Ϸ�
				<< "Incorrect number of weight blobs.";
			// ��weight_shape��Ϊbobs_[0]��shape��������쳣
			if (weight_shape != this->blobs_[0]->shape()) {
				Blob<Dtype> weight_shaped_blob(weight_shape);
				LOG(FATAL) << "Incorrect weight shape: expected shape "
					<< weight_shaped_blob.shape_string() << "; instead, shape was "
					<< this->blobs_[0]->shape_string();
			}
			// ��bias_shape��Ϊbobs_[1]��shape��������쳣
			if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
				Blob<Dtype> bias_shaped_blob(bias_shape);
				LOG(FATAL) << "Incorrect bias shape: expected shape "
					<< bias_shaped_blob.shape_string() << "; instead, shape was "
					<< this->blobs_[1]->shape_string();
			}
			LOG(INFO) << "Skipping parameter initialization";
		}
		// ��blobs_.size() = 0����ô����bias_term_����α����blobs_�Ĵ�С��ʼ��
		else {
			if (bias_term_) {
				this->blobs_.resize(2);
			}
			else {
				this->blobs_.resize(1);
			}
			// Initialize and fill the weights:
			// output channels x input channels per-group x kernel height x kernel width
			// ���ջ�ȡ����״��ʼ��������Ȩֵ����״�����ͨ������ÿ������ͨ����������˸߶ȡ�����˿��
			this->blobs_[0].reset(new Blob<Dtype>(weight_shape));	// ��blobs_[0]��С��ʼ��Ϊweight_shape
			// ����protobuf�����õ��˲���Ȩֵ��ʼ������constant,xavier�ȣ����г�ʼ����
			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.convolution_param().weight_filler()));	// ��ȡ���Ƕ����Ĳ����е�Ȩ����䣬Ĭ��Ϊ0
			weight_filler->Fill(this->blobs_[0].get());
			// If necessary, initialize and fill the biases.
			if (bias_term_) {
				this->blobs_[1].reset(new Blob<Dtype>(bias_shape));	// ��������ƫ�ã����ȡ���Ƕ����Ĳ����е�ƫ����䣬Ĭ��Ϊ0
				shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
					this->layer_param_.convolution_param().bias_filler()));
				bias_filler->Fill(this->blobs_[1].get());	// ����ƫ�õ����
			}
		}
		// ��ȡһ�����ͨ����Ӧ�����о���˶������һ�����������ͨ������һ�δ�����������С��Ϊ(������ͨ����/�������)*����˸�*����˿�
		kernel_dim_ = this->blobs_[0]->count(1);	// �ӵ�һ��ά�ȿ�ʼͳ��Ȩֵ��������ÿ���˲����Ĵ�С��ÿ������ͨ����������˸߶ȡ�����˿��
		weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;	//��ȡȨ�ص�ƫ������д��(conv_out_channels_ / group_) * kernel_dim_��ֱ�ۡ�
		// Propagate gradients to the parameters (as directed by backward pass).
		this->param_propagate_down_.resize(this->blobs_.size(), true);// ��ʼ����Ȩ�غ�ƫ��(��ѡ)�ݶȷ����Ŀ���
	}

	// reshape������Ҫ�ǣ�����bottom��״���������top blob��״������im2col�������һЩ������
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const int first_spatial_axis = channel_axis_ + 1;	// �ҵ������������ĵ�һά��������ͨ��Ϊheight
		/* ��������blob��ά���Ƿ���ھ����������ĵ�һά���������Ͼ��������Ҫ�����ά���� */
		CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
			<< "bottom num_axes may not change.";
		num_ = bottom[0]->count(0, channel_axis_);	// ͳ����������ͼ����batch_size*channel_num������ȡ�������������ͼƬ��Ŀ
		CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)	// ��������ͨ�����Ƿ�Ϸ�
			<< "Input size incompatible with convolution kernel.";
		// TODO: generalize to handle inputs of different shapes. һ�㻯����Ӧ��ͬ��״�����롣
		// �������bottom blob��״һ��
		for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
			CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())	// ���������blob�Ļ����������blob�Ƿ������ͬ��shape
				<< "shape mismatch - bottom[0]: " << bottom[0]->shape_string()
				<< " vs. bottom[" << bottom_id << "]: "
				<< bottom[bottom_id]->shape_string();
		}
		// Shape the tops. ����bottom��״���������top blob��״��batch_size, channel_out, out_h, out_w,...��
		bottom_shape_ = &bottom[0]->shape();	// ��ȡ����������blob����״
		compute_output_shape();	// ��ȡ����������blob����״���麯������Ҫ��д��ȷ�����������״
		// ����[begin,end)��������һ�����飨����bottom��״����Ԫ�ص���vector�У�����ҿ��������blob��b*c*h*w����ֻ���ƹ�����b��
		vector<int> top_shape(bottom[0]->shape().begin(), bottom[0]->shape().begin() + channel_axis_);	// ��ʼ��top_shape��һ��Ԫ��Ϊ���뵥λblob��num
		top_shape.push_back(num_output_);	// top_shape���������ͨ����
		// ���õ���top_shape����top blob������������״��Ϊ�俪�ٿռ�ȡ�
		for (int i = 0; i < num_spatial_axes_; ++i) {
			top_shape.push_back(output_shape_[i]);	// top_shape�����������ά��
		}
		// ��channel��ʼͳ���������ͼ��С��h*w*...
		for (int top_id = 0; top_id < top.size(); ++top_id) {
			top[top_id]->Reshape(top_shape);	// ��top��ÿ��blob���г�ʼ��
		}
		/* ���Ҫ��ת���������conv_out_spatial_dim_��ʼ��Ϊ����������λblob(bottom[0])�ĵ�ͨ���������� */
		if (reverse_dimensions()) {
			conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
		}
		/* ����conv_out_spatial_dim_��ʼ��Ϊ����������λblob(top[0])�ĵ�ͨ���������� */
		else {
			conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
		}
		// ������������롰ͼ���ϰ������������γ��˶����ͼ��Ȼ��������ͼ����һ�У��еĳ��Ⱦ���col_offset_��
		// col_offset_��im2col_cpu()������channels_col�ļ��������Ƶģ�����ֵ������ȣ�
		// ԭ�����ڣ�channels_col�ǽ�����������ͨ����conv_in_channels_������ˣ�
		// ��kernel_dim_ֻ�õ���һ����channel,��conv_in_channels_/group_��
		col_offset_ = kernel_dim_ * conv_out_spatial_dim_;	// col_offset������һ�����ͨ����Ӧ�����о���˴����һ������������������
		// �������������ͼҲҪ���飬��Ȼgroup_Ĭ��Ϊ1��д��(conv_out_channels_ / group_) * conv_out_spatial_dim_��ֱ��
		output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;	// output_offset_������һ����������������������
		// Setup input dimensions (conv_input_shape_).
		vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);	// ���ڳ�ʼ����������������ݵ���״��һ����ά(C,H,W)
		conv_input_shape_.Reshape(bottom_dim_blob_shape);	// ��ʼ�����������shape��һ���СΪ3
		int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
		// ��ʼ�����������������һ��˳��Ϊchannel->height->width
		for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
			if (reverse_dimensions()) {
				conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
			}
			else {
				conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
			}
		}
		// The im2col result buffer will only hold one image at a time to avoid
		// overly large memory usage. In the special case of 1x1 convolution
		// it goes lazily unused to save memory.
		// ÿ��im2colֻת��һ��ͼ��
		col_buffer_shape_.clear();
		col_buffer_shape_.push_back(kernel_dim_ * group_);	// col_buffer_shape_����(������ͨ����*����˸�*����˿�)
		// col_buffer_shape_�������������ͨ����ά��
		for (int i = 0; i < num_spatial_axes_; ++i) {
			if (reverse_dimensions()) {
				col_buffer_shape_.push_back(input_shape(i + 1));
			}
			else {
				col_buffer_shape_.push_back(output_shape_[i]);
			}
		}
		// ������Ϊcol_buffer_�����洢�����ݵ�ά��Ϊ��(kernel_dim_ * group_) �� H �� W.
		col_buffer_.Reshape(col_buffer_shape_);	// ��ʼ��col_buffer
		bottom_dim_ = bottom[0]->count(channel_axis_);	// bottom_dim_��������bottom blob��һ��channel������������
		top_dim_ = top[0]->count(channel_axis_);	// top_dim_��������top blob��һ��channel������������
		num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;	// ������һ�����ͨ����Ӧ�����о���˶�ȫ���������������ʱת�����ɵ�������������
		num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;	// �����˽����ɵ���������ԭ�������������ͼ������
		// Set up the all ones "bias multiplier" for adding biases by BLAS
		out_spatial_dim_ = top[0]->count(first_spatial_axis);	// ����������ĵ�ͨ��������
		// ��������ƫ�ã���ô��ʼ��ƫ�ó���blob
		if (bias_term_) {
			// ƫ�ó����Ĵ�СΪ����ĵ�ͨ������������Ϊ����ÿ��������ݳ�����һ��
			vector<int> bias_multiplier_shape(1, out_spatial_dim_);
			bias_multiplier_.Reshape(bias_multiplier_shape);
			caffe_set(bias_multiplier_.count(), Dtype(1),	// �Ƚ���Щ������Ϊ1
				bias_multiplier_.mutable_cpu_data());
		}
	}

	// ǰ�򡢷��򴫲�������¾���������im2col����չ�ɾ��󣬵���cblas_sgemm�������о���˷����㣬ʵ�־��������
	template <typename Dtype>
	// �������ݵ�cpuǰ�򴫲�
	void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col) {
		const Dtype* col_buff = input;
		// ���û��1x1�����Ҳû��skip_im2col
		// ��ʹ��conv_im2col_cpu��ʹ�þ���˻��������е�ÿһ��kernel��С��ͼ���
		// ���һ�����������γ�һ��height=kernel_dim_��
		// width = �����ͼ��height*�����ͼ��width
		if (!is_1x1_) {
			// im2col��һ��������������ԭ����ͼ��С����ɲ���������
			if (!skip_im2col) {
				conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
			}
			col_buff = col_buffer_.cpu_data();
		}
		// caffe_cpu_gemm���ǵ���cblas_sgemm�������о���˷�����
		for (int g = 0; g < group_; ++g) {
			// conv_out_channels_ / group_��ÿ�������������channel
			// kernel_dim_ = input channels per-group x kernel height x kernel width
			// �������output[output_offset_ * g]= weights[weight_offset_ * g] X col_buff[col_offset_ * g]
			// weights��ά��Ϊ(conv_out_channels_ / group_) x kernel_dim_
			// weights����״�� [conv_out_channel x kernel_dim_]
			// col_buff�൱�����ݣ�������״��[kernel_dim_ x (�����ͼ��߶�*�����ͼ����)]=
			//    kernel_dim_ x conv_out_spatial_dim_
			// ����output����״��Ȼ����conv_out_channel X (�����ͼ��߶�*�����ͼ����)=
			//    (conv_out_channels_ /group_) x conv_out_spatial_dim_
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
				group_, conv_out_spatial_dim_, kernel_dim_,
				(Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
				(Dtype)0., output + output_offset_ * g);
		}
	}

	// ���������ݶȵ�cpu���򴫲���������bias
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output, const Dtype* bias) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
			out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
			(Dtype)1., output);
	}

	// ���������ݶȵ�cpu���򴫲����������bottom data�ĵ����Ա㴫����һ��
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input) {
		Dtype* col_buff = col_buffer_.mutable_cpu_data();
		if (is_1x1_) {
			col_buff = input;
		}
		for (int g = 0; g < group_; ++g) {
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
				conv_out_spatial_dim_, conv_out_channels_ / group_,
				(Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
				(Dtype)0., col_buff + col_offset_ * g);
		}
		if (!is_1x1_) {
			conv_col2im_cpu(col_buff, input);	// �����е���������ԭ��ͼ��
		}
	}

	// ����Ȩ�ص�cpuǰ�򴫲����������weight�ĵ������ڸ��¡�
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights) {
		const Dtype* col_buff = input;
		if (!is_1x1_) {
			conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
			col_buff = col_buffer_.cpu_data();
		}
		for (int g = 0; g < group_; ++g) {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
				kernel_dim_, conv_out_spatial_dim_,
				(Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
				(Dtype)1., weights + weight_offset_ * g);
		}
	}

	// ����ƫ���ݶȵ�cpu���򴫲����������bias�ĵ���
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias, const Dtype* input) {
		caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
			input, bias_multiplier_.cpu_data(), 1., bias);
	}

// ��Ϊ����CPU_ONLY����ʹ��GPU
#ifndef CPU_ONLY
	// �������ݵ�gpuǰ�򴫲�
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
		const Dtype* weights, Dtype* output, bool skip_im2col) {
		const Dtype* col_buff = input;
		if (!is_1x1_) {
			if (!skip_im2col) {
				conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
			}
			col_buff = col_buffer_.gpu_data();
		}
		for (int g = 0; g < group_; ++g) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
				group_, conv_out_spatial_dim_, kernel_dim_,
				(Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
				(Dtype)0., output + output_offset_ * g);
		}
	}

	// ����ƫ�õ�gpuǰ�򴫲�
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
		const Dtype* bias) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
			out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
			(Dtype)1., output);
	}

	// ���������ݶȵ�gpu���򴫲�
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
		const Dtype* weights, Dtype* input) {
		Dtype* col_buff = col_buffer_.mutable_gpu_data();
		if (is_1x1_) {
			col_buff = input;
		}
		for (int g = 0; g < group_; ++g) {
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
				conv_out_spatial_dim_, conv_out_channels_ / group_,
				(Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
				(Dtype)0., col_buff + col_offset_ * g);
		}
		if (!is_1x1_) {
			conv_col2im_gpu(col_buff, input);
		}
	}

	// ����Ȩ�ص�gpuǰ�򴫲�
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
		const Dtype* output, Dtype* weights) {
		const Dtype* col_buff = input;
		if (!is_1x1_) {
			conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
			col_buff = col_buffer_.gpu_data();
		}
		for (int g = 0; g < group_; ++g) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
				kernel_dim_, conv_out_spatial_dim_,
				(Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
				(Dtype)1., weights + weight_offset_ * g);
		}
	}

	// ����ƫ���ݶȵķ��򴫲�
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
		const Dtype* input) {
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
			input, bias_multiplier_.gpu_data(), 1., bias);
	}

#endif  // !CPU_ONLY

	INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
