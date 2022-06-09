//
// Created by Muyu Li on 27.06.22.
//

#ifndef ML_INPUT_DATA_SERVICE_SPLIT_MARKER_OP_H
#define ML_INPUT_DATA_SERVICE_SPLIT_MARKER_OP_H

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace easl{

class SplitMarkerOp : public UnaryDatasetOpKernel {
public:
    static constexpr const char* const kDatasetType = "SplitMarkerDataset";

    explicit SplitMarkerOp(OpKernelConstruction* ctx);

protected:
    void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                     DatasetBase** output) override;

private:
    class Dataset;
};

} // namespace easl
} // namespace experimental
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_SPLIT_MARKER_OP_H
