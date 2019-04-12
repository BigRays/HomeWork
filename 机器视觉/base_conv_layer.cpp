#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

//// 卷积核的形状[kernel_h, kernel_w]
// Blob<int> kernel_shape_;

//// 步长形状[stride_h, stride_w]
// Blob<int> stride_;

//// padding形状[pad_h, pad_w]
// Blob<int> pad_;

//// 扩张卷积的形状，就是镂空式的卷积
// Blob<int> dilation_;

//// 卷积的输入形状 = [输入图像通道数, 输入图像h, 输入图像w]
// Blob<int> conv_input_shape_;

//// col_buffer的形状 = [kernel_dim_, conv_out_spatial_dim_ ]
//// 即将图像转化成利于卷积的展开体col形式（具体参考src/utils/im2col.cpp），存于col_buffer，将卷积核权值×col_buffer=卷积输出，所以其形状为上述样子。
// vector<int> col_buffer_shape_;

//// 层输出的形状，存在vector里
// vector<int> output_shape_;

//// 层输入的形状，存在vector里，返回指针，因为是别的层的输出，直接用指针指向之前已经存在的上一层的output_shape_。
// const vector<int>* bottom_shape_;

//// 空间轴个数，就是输入是几维图像
// int num_spatial_axes_;

//// 输入度维度 = 输入通道数*输入图像的h*输入图像的w
// int bottom_dim_;

//// 输出维度 = 输出通道数*输出图像的h*输出图像的w
// int top_dim_;

//// 输入图像的哪个axis是channel,一般是第二个维度
// int channel_axis_;

//// 堆大小
// int num_;

//// 通道数
// int channels_;

//// 卷积组的大小
// int group_;

//// 输出空间维度 = 卷积之后的图像长*卷积之后图像的宽
// int out_spatial_dim_;

//// 使用卷积组用到的权值偏置
// int weight_offset_;

//// 卷积后的图像的通道数
// int num_output_;

//// 是否启用偏置
// bool bias_term_;

//// 是否是1x1卷积
// bool is_1x1_;

//// 强制使用n维通用卷积，即im2col的n维形式，而不是更常用的二维形式。
// bool force_nd_im2col_;

// im2col：图像矩阵的卷积运算通过im2col转化为高效的矩阵乘法运算。
// 简单来讲，图像的卷积操作，就是滤波器算子逐个区域扫描图像的过程，在每个区域内都是滤波器权值和图像块像素值之间的简单的数乘
// 因此，可以把每个图像块展成一列，将滤波器展成一行，进行矩阵乘法，实现卷积操作

// CHECK_EQ(x, y) << "x!=y"，EQ即equation，意为“等于”，函数判断是否x等于y，当x != y时，函数打印出x != y。
//
// CHECK_NE(x, y) << "x=y"，NE即not equation，意为“不等于”，函数判断是否x不等于y，当x = y时，函数打印出x = y。
//
// CHECK_LE(x, y) << "x<=y", LE即lower equation, 意为小于等于，函数判断是否x小于等于y。当x <= y时，函数打印x <= y。
//
// CHECK_LT(x, y) << "x<y", LT即为lower to ，意为小于，函数判断是否x小于y，当x < y时，函数打印x < y。
//
// CHECK_GE(x, y) << "x>=y", GE即为great equation，意为大于等于，函数判断是否x大于等于y。当x >= y时，函数打印x >= y。
//
// CHECK_GT(x, y) << "x>y", GT即为great to ，意为大于，函数判断是否x大于y，当x > y时，函数打印x > y。


// base_conv_layer.cpp 中定义了 BaseConvolutionLayer 类的一些成员函数，而 BaseConvolutionLayer 是 ConvolutionLayer 的父类
// ConvolutionLayer 中用到的一些函数都在这里定义，所以在看 conv_layer 前需要看此源代码。
namespace caffe {
	// 这个函数主要是根据 protobuf 中的层参数设置，配置卷积核的大小，padding，步长和输入、权值参数初始化等等。是层建立的基础，是大多层建立的第一步，层参数设置及初始化。
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// Configure the kernel size, padding, stride, and inputs.
		// 根据protobuf中的层参数设置，配置卷积核的大小，padding，步长和输入等等。
		ConvolutionParameter conv_param = this->layer_param_.convolution_param();	// 读入参数
		force_nd_im2col_ = conv_param.force_nd_im2col();	// 根据层参数设置是否强制进行n维im2col
		// 下面的函数定义于blob.hpp，用于检查输入index是否在范围内，并支持输入负数反向索引，输出正数正向索引，
		// 确定哪个维度是channel，一般是，batch*channel*height*width，第二个维度。
		/* channel_axis_这个参数读取参数定义中的axis参数，默认为1，表示按channel求和，输入blob为(N,C,W,H)时，
		一个输出通道对应的所有卷积核对输入blob上各通道做二维卷积，最后将输入各通道卷积的结果加起来，作为
		一张输出的特征子图 */
		channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
		// channel后紧跟的就是图像高、宽这种空间轴，第一个空间轴是第几维自然就是+1
		const int first_spatial_axis = channel_axis_ + 1;	// 指示卷积输入图像的第一个轴，往往是H(height)
		const int num_axes = bottom[0]->num_axes();	// 得到bottom blob的维度
		num_spatial_axes_ = num_axes - first_spatial_axis;	// 空间轴个数，卷积处理的维度数
		CHECK_GE(num_spatial_axes_, 0);	// 卷积处理的维度数必须大于0

		vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);	// 用于初始化卷积操作输入数据的形状，一般三维(C,H,W)
		// 当num_spatial_axes_==2时，spatial_dim_blob_shape这个vector只包含一个元素且值为2
		vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));	//用于初始化卷积核的形状
		// 配置滤波器卷积核维数
		// Setup filter kernel dimensions (kernel_shape_).
		// 调用blob.cpp里的 void Blob<Dtype>::Reshape(const vector<int>& shape)
		// 以spatial_dim_blob_shape为参数来构造一个Blob，即kernel_shape_，则这个Blob的维度信息只包含一个维度，值为2,
		// 也就是说这个Blob的count_==2。尽管这个Blob的维度信息只包含一个维度,只有两个数。
		// 因为在后续的计算（Im2col）中，我只关心这个Blob中的数据的值，而不关心这个Blob的shape信息。
		// 例如在Im2col()中，只要取出相应数值即可kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1]。
		kernel_shape_.Reshape(spatial_dim_blob_shape);	// 初始化卷积核的形状(高*宽)
		int* kernel_shape_data = kernel_shape_.mutable_cpu_data();	// 得到记录卷积核形状数据地址
		// 若层参数中有定义卷积核高或宽，则将其存入kernel_shape_data这一blob中
		/* 检查参数中有没有自定义二维卷积的卷积核长宽，如果有定义则分别赋值，且自定义了二维卷积核
		长宽的话，kernal_size参数将不能被定义，否则非法。若参数中没有定义二维卷积核的长宽，那么根据
		kernal_size参数给卷积核赋值，卷积核一般是正方形 */
		if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
			CHECK_EQ(num_spatial_axes_, 2)
				<< "kernel_h & kernel_w can only be used for 2D convolution.";
			CHECK_EQ(0, conv_param.kernel_size_size())
				<< "Either kernel_size or kernel_h/w should be specified; not both.";
			// 卷积核的高
			kernel_shape_data[0] = conv_param.kernel_h();
			// 卷积核的宽
			kernel_shape_data[1] = conv_param.kernel_w();
		}
		// 若层参数中没有定义卷积核宽和高，则根据卷积核的维度数来确定。哪一维核大小就附几
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
		// 检查卷积核参数(高宽)是否合法
		for (int i = 0; i < num_spatial_axes_; ++i) {
			CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
		}
		// 接下来的stride_，pad_，dilation_这些blob的设定也类似。
		// 配置步长维度
		// Setup stride dimensions (stride_)
		stride_.Reshape(spatial_dim_blob_shape);
		int* stride_data = stride_.mutable_cpu_data();
		/* 检查参数中有没有自定义二维卷积时高和宽方向的步长，如果定义了则赋值。如果没有定义的话，就按照我们
		定义的网络参数文件中的卷积层的stride参数赋值，stride参数要是缺失的话步长默认为kDefaultStride，即为1，
		我们往往只定义了一个步长值，代表高和宽方向的步长一致。 */
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
		/* 检查参数中有没有自定义高和宽方向的pad，如果定义了则赋值。如果没有定义的话，就按照我们
		定义的网络参数文件中的卷积层的pad参数赋值，pad参数要是缺失的话默认为kDefaultPad，即为0，
		我们往往只定义了一个pad值，代表高和宽方向的pad一致。 */
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
		/* 检查参数中有没有自定义高和宽方向的卷积核扩展，如果定义了则赋值。如果没有定义的话，就按照我们
		定义的网络参数文件中的卷积层的dilation参数赋值，dilation_参数要是缺失的话默认为kDefaultDilation，
		即为1，表示卷积核不进行扩展。 */
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
		// 判断是不是1*1卷积
		is_1x1_ = true;
		for (int i = 0; i < num_spatial_axes_; ++i) {
			is_1x1_ &=
				kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
			if (!is_1x1_) { break; }
		}
		// Configure output channels and groups.
		channels_ = bottom[0]->shape(channel_axis_);	// 根据channel的维度索引找数据blob的channel数。获取卷积层输入的单blob的通道数。
		num_output_ = this->layer_param_.convolution_param().num_output();	// 获取卷积层输出的通道数
		CHECK_GT(num_output_, 0);	// 核验输出通道数是否大于零
		group_ = this->layer_param_.convolution_param().group();	// 获取卷积组大小
		// 输入、输出 channel 个数必须为group的整数倍，每个group中的卷积核只对本group对应频道的特征图进行卷积操作
		CHECK_EQ(channels_ % group_, 0);	// 核验输入的单blob通道数是否能被卷积组数整除
		CHECK_EQ(num_output_ % group_, 0)	// 核验输出通道数是否能被卷积组数整除
			<< "Number of output should be multiples of group.";
		// 若需要反转卷积操作，则交换输入输出，否则不交换
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
		// 权值形状：（conv_out_channels_，conv_in_channels_ / group_，kernel_h, kernel_w）,如：256*256*3*3
		vector<int> weight_shape(2);	// 定义卷积层参数规格
		weight_shape[0] = conv_out_channels_;	// 权重参数shape的第一个数为输出通道大小，即每个输出通道对应各自的卷积核，理解为num
		// 每个group中的卷积核只对本group对应频道的特征图进行卷积操作，即上文中描述的3x3x32和3x3x16的区别（group的影响）
		weight_shape[1] = conv_in_channels_ / group_;	// 权重参数shape的第二个数为输入通道大小除以卷积组数，理解为channel
		for (int i = 0; i < num_spatial_axes_; ++i) {
			weight_shape.push_back(kernel_shape_data[i]);	// 权重参数shape的第三个和第四个数为卷积核维度大小
		}
		bias_term_ = this->layer_param_.convolution_param().bias_term();	// 获取是否使用偏置的参数
		vector<int> bias_shape(bias_term_, num_output_);	// 定义偏置参数规格，若bias_term_为true(1)，那么bias_shape[0]=num_output_
		if (this->blobs_.size() > 0) {
			CHECK_EQ(1 + bias_term_, this->blobs_.size())	// 核验blobs_是否合法
				<< "Incorrect number of weight blobs.";
			// 若weight_shape不为bobs_[0]的shape，则输出异常
			if (weight_shape != this->blobs_[0]->shape()) {
				Blob<Dtype> weight_shaped_blob(weight_shape);
				LOG(FATAL) << "Incorrect weight shape: expected shape "
					<< weight_shaped_blob.shape_string() << "; instead, shape was "
					<< this->blobs_[0]->shape_string();
			}
			// 若bias_shape不为bobs_[1]的shape，则输出异常
			if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
				Blob<Dtype> bias_shaped_blob(bias_shape);
				LOG(FATAL) << "Incorrect bias shape: expected shape "
					<< bias_shaped_blob.shape_string() << "; instead, shape was "
					<< this->blobs_[1]->shape_string();
			}
			LOG(INFO) << "Skipping parameter initialization";
		}
		// 若blobs_.size() = 0，那么根据bias_term_的真伪进行blobs_的大小初始化
		else {
			if (bias_term_) {
				this->blobs_.resize(2);
			}
			else {
				this->blobs_.resize(1);
			}
			// Initialize and fill the weights:
			// output channels x input channels per-group x kernel height x kernel width
			// 按照获取的形状初始化并填入权值，形状：输出通道数×每组输入通道数×卷积核高度×卷积核宽度
			this->blobs_[0].reset(new Blob<Dtype>(weight_shape));	// 将blobs_[0]大小初始化为weight_shape
			// 根据protobuf里设置的滤波器权值初始方法（constant,xavier等）进行初始化。
			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.convolution_param().weight_filler()));	// 读取我们定义层的参数中的权重填充，默认为0
			weight_filler->Fill(this->blobs_[0].get());
			// If necessary, initialize and fill the biases.
			if (bias_term_) {
				this->blobs_[1].reset(new Blob<Dtype>(bias_shape));	// 若启用了偏置，则读取我们定义层的参数中的偏置填充，默认为0
				shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
					this->layer_param_.convolution_param().bias_filler()));
				bias_filler->Fill(this->blobs_[1].get());	// 进行偏置的填充
			}
		}
		// 获取一个输出通道对应的所有卷积核对输入的一个卷积组所有通道操作一次处理数据量大小，为(输入总通道数/卷积组数)*卷积核高*卷积核宽
		kernel_dim_ = this->blobs_[0]->count(1);	// 从第一个维度开始统计权值数量，即每个滤波器的大小：每组输入通道数×卷积核高度×卷积核宽度
		weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;	//获取权重的偏移量，写成(conv_out_channels_ / group_) * kernel_dim_更直观。
		// Propagate gradients to the parameters (as directed by backward pass).
		this->param_propagate_down_.resize(this->blobs_.size(), true);// 初始化对权重和偏置(可选)梯度反传的开关
	}

	// reshape函数主要是：根据bottom形状，计算输出top blob形状；计算im2col及反向的一些参数。
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const int first_spatial_axis = channel_axis_ + 1;	// 找到卷积操作处理的第一维的索引，通常为height
		/* 核验输入blob的维度是否等于卷积操作处理的第一维的索引加上卷积操作需要处理的维度数 */
		CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
			<< "bottom num_axes may not change.";
		num_ = bottom[0]->count(0, channel_axis_);	// 统计输入特征图数量batch_size*channel_num，即获取卷积层操作输入的图片数目
		CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)	// 检查输入的通道数是否合法
			<< "Input size incompatible with convolution kernel.";
		// TODO: generalize to handle inputs of different shapes. 一般化来适应不同形状的输入。
		// 检查所有bottom blob形状一致
		for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
			CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())	// 如果输入多个blob的话，检查所有blob是否具有相同的shape
				<< "shape mismatch - bottom[0]: " << bottom[0]->shape_string()
				<< " vs. bottom[" << bottom_id << "]: "
				<< bottom[bottom_id]->shape_string();
		}
		// Shape the tops. 根据bottom形状，计算输出top blob形状（batch_size, channel_out, out_h, out_w,...）
		bottom_shape_ = &bottom[0]->shape();	// 获取卷积层输入的blob的形状
		compute_output_shape();	// 获取卷积层输出的blob的形状，虚函数，需要重写来确定具体输出形状
		// 复制[begin,end)区间内另一个数组（描述bottom形状）的元素到该vector中，左闭右开，如果是blob是b*c*h*w，就只复制过来了b。
		vector<int> top_shape(bottom[0]->shape().begin(), bottom[0]->shape().begin() + channel_axis_);	// 初始化top_shape第一个元素为输入单位blob的num
		top_shape.push_back(num_output_);	// top_shape加入输出的通道数
		// 按得到的top_shape创建top blob，并调整其形状，为其开辟空间等。
		for (int i = 0; i < num_spatial_axes_; ++i) {
			top_shape.push_back(output_shape_[i]);	// top_shape加入卷积处理的维度
		}
		// 从channel后开始统计输出特征图大小，h*w*...
		for (int top_id = 0; top_id < top.size(); ++top_id) {
			top[top_id]->Reshape(top_shape);	// 将top的每个blob进行初始化
		}
		/* 如果要反转卷积操作，conv_out_spatial_dim_初始化为卷积层输出单位blob(bottom[0])的单通道的数据量 */
		if (reverse_dimensions()) {
			conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
		}
		/* 否则，conv_out_spatial_dim_初始化为卷积层输出单位blob(top[0])的单通道的数据量 */
		else {
			conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
		}
		// 卷积窗口在输入“图像”上按步长滑动，形成了多个子图；然后将所有子图拉成一列，列的长度就是col_offset_。
		// col_offset_与im2col_cpu()函数中channels_col的计算是相似的，但是值并不相等，
		// 原因在于：channels_col是将卷积层输入的通道数conv_in_channels_用于相乘，
		// 但kernel_dim_只用到了一部分channel,即conv_in_channels_/group_。
		col_offset_ = kernel_dim_ * conv_out_spatial_dim_;	// col_offset表征了一个输出通道对应的所有卷积核处理的一个卷积组的所有数据量
		// 卷积层的输出特征图也要分组，当然group_默认为1。写成(conv_out_channels_ / group_) * conv_out_spatial_dim_更直观
		output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;	// output_offset_表征了一个卷积组输出的所有数据量
		// Setup input dimensions (conv_input_shape_).
		vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);	// 用于初始化卷积操作输入数据的形状，一般三维(C,H,W)
		conv_input_shape_.Reshape(bottom_dim_blob_shape);	// 初始化卷积层输入shape，一般大小为3
		int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
		// 初始化卷积层的输入参数，一般顺序为channel->height->width
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
		// 每次im2col只转化一张图。
		col_buffer_shape_.clear();
		col_buffer_shape_.push_back(kernel_dim_ * group_);	// col_buffer_shape_加入(输入总通道数*卷积核高*卷积核宽)
		// col_buffer_shape_加入卷积层输出单通道的维度
		for (int i = 0; i < num_spatial_axes_; ++i) {
			if (reverse_dimensions()) {
				col_buffer_shape_.push_back(input_shape(i + 1));
			}
			else {
				col_buffer_shape_.push_back(output_shape_[i]);
			}
		}
		// 可以认为col_buffer_内所存储的数据的维度为：(kernel_dim_ * group_) × H × W.
		col_buffer_.Reshape(col_buffer_shape_);	// 初始化col_buffer
		bottom_dim_ = bottom[0]->count(channel_axis_);	// bottom_dim_描述的是bottom blob的一个channel包含的数据量
		top_dim_ = top[0]->count(channel_axis_);	// top_dim_描述的是top blob的一个channel包含的数据量
		num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;	// 描述了一个输出通道对应的所有卷积核对全部输入做卷积操作时转换生成的列向量的数量
		num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;	// 描述了将生成的列向量还原卷积操作的区域图的数量
		// Set up the all ones "bias multiplier" for adding biases by BLAS
		out_spatial_dim_ = top[0]->count(first_spatial_axis);	// 描述了输出的单通道数据量
		// 若启用了偏置，那么初始化偏置乘数blob
		if (bias_term_) {
			// 偏置乘数的大小为输出的单通道数据量，因为对于每个输出数据乘数不一样
			vector<int> bias_multiplier_shape(1, out_spatial_dim_);
			bias_multiplier_.Reshape(bias_multiplier_shape);
			caffe_set(bias_multiplier_.count(), Dtype(1),	// 先将这些乘数置为1
				bias_multiplier_.mutable_cpu_data());
		}
	}

	// 前向、反向传播计算大致就两步：用im2col函数展成矩阵，调用cblas_sgemm函数进行矩阵乘法运算，实现卷积操作。
	template <typename Dtype>
	// 进行数据的cpu前向传播
	void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col) {
		const Dtype* col_buff = input;
		// 如果没有1x1卷积，也没有skip_im2col
		// 则使用conv_im2col_cpu对使用卷积核滑动过程中的每一个kernel大小的图像块
		// 变成一个列向量，形成一个height=kernel_dim_的
		// width = 卷积后图像height*卷积后图像width
		if (!is_1x1_) {
			// im2col将一个卷积操作处理的原特征图按小窗变成并排列向量
			if (!skip_im2col) {
				conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
			}
			col_buff = col_buffer_.cpu_data();
		}
		// caffe_cpu_gemm就是调用cblas_sgemm函数进行矩阵乘法运算
		for (int g = 0; g < group_; ++g) {
			// conv_out_channels_ / group_是每个卷积组的输出的channel
			// kernel_dim_ = input channels per-group x kernel height x kernel width
			// 计算的是output[output_offset_ * g]= weights[weight_offset_ * g] X col_buff[col_offset_ * g]
			// weights的维度为(conv_out_channels_ / group_) x kernel_dim_
			// weights的形状是 [conv_out_channel x kernel_dim_]
			// col_buff相当于数据，它的形状是[kernel_dim_ x (卷积后图像高度*卷积后图像宽度)]=
			//    kernel_dim_ x conv_out_spatial_dim_
			// 所以output的形状自然就是conv_out_channel X (卷积后图像高度*卷积后图像宽度)=
			//    (conv_out_channels_ /group_) x conv_out_spatial_dim_
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
				group_, conv_out_spatial_dim_, kernel_dim_,
				(Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
				(Dtype)0., output + output_offset_ * g);
		}
	}

	// 进行数据梯度的cpu反向传播，卷积后加bias
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output, const Dtype* bias) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
			out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
			(Dtype)1., output);
	}

	// 进行数据梯度的cpu反向传播，计算关于bottom data的导数以便传给下一层
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
			conv_col2im_cpu(col_buff, input);	// 将并列的列向量还原成图像
		}
	}

	// 进行权重的cpu前向传播，计算关于weight的导数用于更新。
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

	// 进行偏置梯度的cpu反向传播，计算关于bias的导数
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias, const Dtype* input) {
		caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
			input, bias_multiplier_.cpu_data(), 1., bias);
	}

// 若为定义CPU_ONLY，则使用GPU
#ifndef CPU_ONLY
	// 进行数据的gpu前向传播
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

	// 进行偏置的gpu前向传播
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
		const Dtype* bias) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
			out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
			(Dtype)1., output);
	}

	// 进行数据梯度的gpu反向传播
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

	// 进行权重的gpu前向传播
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

	// 进行偏置梯度的反向传播
	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
		const Dtype* input) {
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
			input, bias_multiplier_.gpu_data(), 1., bias);
	}

#endif  // !CPU_ONLY

	INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
