//
// Created by Muyu Li on 21.05.22.
//

#ifndef ML_INPUT_DATA_SERVICE_SPLIT_SECOND_HALF_H
#define ML_INPUT_DATA_SERVICE_SPLIT_SECOND_HALF_H

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace split_second_half {

class SplitSecondHalfOp: public UnaryDatasetOpKernel {
public:
    static constexpr const char* const kDatasetType = "SplitSecondHalf";

    explicit SplitSecondHalfOp(OpKernelConstruction* ctx);

protected:
    void MakeDataset(OpKernelContext* ctx, DatasetBase* input, DatasetBase** output) override;

//private:
//    void rewrite(GraphDef* dsdo_graph, // data_service_dataset_op
//                 std::string dsdo_sink_node_name,
//                 GraphDef* second_half_graph,
//                 std::string second_half_sink_node_name,
//                 int64 split_node_index);
//
//    NodeDef* addNodeToGraph(NodeDef* node, MutableGraphView& graph_view);
//
//    NodeDef* getNodeDefFromName(std::string name, GraphDef* graph);
//
//    Status BFSGraph(std::string sink_node_name,
//                    GraphDef* output,
//                    std::string prefix);
};

//class SplitSecondHalfOp::Dataset: public DatasetBase {
//public:
//    Dataset(OpKernelContext* ctx, const DatasetBase* input);
//
//    Dataset(DatasetContext::Params params, const DatasetBase* input);
//
//    ~Dataset() override;
//
//    std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const override;
//
//    const DataTypeVector& output_dtypes() const override;
//
//    const std::vector<PartialTensorShape>& output_shapes() const override;
//
//    string DebugString() const override;
//
//    Status CheckExternalState() const override;
//
//protected:
//    Status AsGraphDefInternal(SerializationContext* ctx,
//                              DatasetGraphDefBuilder* b,
//                              Node** output) const override;
//
//private:
//    class Iterator;
//    const DatasetBase* const input_;
//};
//
//class SplitSecondHalfOp::Dataset::Iterator: public DatasetIterator<Dataset> {
//public:
//    explicit Iterator(const Params& params): DatasetIterator<Dataset>(params) {}
//
//    Status Initialize(IteratorContext* ctx) override {
//      return Status::OK();
//    }
//
//    Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors, bool* end_of_sequence) override;
//
//    Status SaveInternal(SerializationContext* ctx,
//                        IteratorStateWriter* writer) override {
//      return Status::OK();
//    }
//
//    Status RestoreInternal(IteratorContext* ctx,
//                           IteratorStateReader* reader) override {
//      return Status::OK();
//    }
//};


} // split
} // experimental
} // data
} // tensorflow


#endif //ML_INPUT_DATA_SERVICE_SPLIT_SECOND_HALF_H
