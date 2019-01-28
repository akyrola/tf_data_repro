 
#include <utility>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace data {
namespace {
// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following ops.

class DummyDatasetOp : public DatasetOpKernel {
 public:
    explicit DummyDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("n", &n_));
    }
    void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
        *output = new DummyDataset(ctx, n_);
    }

 private:
    class DummyDataset : public DatasetBase {
     public:
        DummyDataset(OpKernelContext* ctx, int n): DatasetBase(DatasetContext(ctx)), n_(n) {
            shapes_ = std::vector<tensorflow::PartialTensorShape>(n_, tensorflow::PartialTensorShape());
            types_ = tensorflow::DataTypeVector(n_, tensorflow::DT_INT32);
        }

        std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const override {
            return std::unique_ptr<IteratorBase>(
                new Iterator({this, strings::StrCat(prefix, "::Dummy")}));
        }

        const DataTypeVector& output_dtypes() const override { return types_; }

        const std::vector<PartialTensorShape>& output_shapes() const override {
            return shapes_;
        }

        string DebugString() const override { return "DummyDatasetOp::Dataset"; }

     protected:
        Status AsGraphDefInternal(SerializationContext* ctx, DatasetGraphDefBuilder* b,
                                  Node** output) const override {
            return Status::OK();
        }

     private:
        class Iterator : public DatasetIterator<DummyDataset> {
         public:
            explicit Iterator(const Params& params) : DatasetIterator<DummyDataset>(params) {}

            Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                                   bool* end_of_sequence) override {
                int row = row_idx_++;
                TensorShape shape;


                // return n fields of integers
                for (int i = 0; i < dataset()->n_; ++i) {
                    tensorflow::Tensor tensor(ctx->allocator({}), tensorflow::DT_INT32, shape);
                    tensor.scalar<tensorflow::int32>()() = row;
                    out_tensors->emplace_back(std::move(tensor));
                }
                return Status::OK();
            }

         private:
            int row_idx_ = 0;
        };

        int n_;

        std::vector<tensorflow::PartialTensorShape>  shapes_;
        tensorflow::DataTypeVector types_;
    };

    int n_;
};

REGISTER_OP("DummyDataset")
    .Output("handle: variant")
    .Attr("n: int")
    .SetIsStateful();

REGISTER_KERNEL_BUILDER(Name("DummyDataset").Device(DEVICE_CPU), DummyDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
