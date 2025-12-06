import argparse
import json
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="pet_mobilenet", help="Name of the registered model")
    parser.add_argument("--output_onnx", default="best_model.onnx", help="Output path for ONNX model")
    parser.add_argument("--labels_out", default="labels.json", help="Output path for labels JSON")
    args = parser.parse_args()

    client = MlflowClient()
    
    versions = client.search_model_versions(f"name='{args.model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{args.model_name}'")

    best_run = None
    best_val_acc = -1.0

    for version in versions:
        run = client.get_run(version.run_id)
        val_acc = run.data.metrics.get("final_val_acc", -1.0)
        
        print(f"Checking version {version.version} (run_id={version.run_id}): acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_run = run

    if not best_run:
        raise RuntimeError("Could not determine best model version (no valid metrics found).")

    print(f"Best model found: run_id={best_run.info.run_id}, acc={best_val_acc:.4f}")

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        args.output_onnx,
        opset_version=18,
        input_names=["input"],
        output_names=["logits"]
    )
    print(f"Exported ONNX model to {args.output_onnx}")

    local_labels_path = client.download_artifacts(run_id=best_run.info.run_id, path="labels.json")
    with open(local_labels_path, "r") as src, open(args.labels_out, "w") as dst:
        json.dump(json.load(src), dst, indent=2)

    print(f"Saved labels to {args.labels_out}")

if __name__ == "__main__":
    main()
