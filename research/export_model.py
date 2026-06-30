import torch
import torchvision.models as models
import torch.nn as nn
import os

def load_model(weights_path):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def export_to_torchscript_mobile(model, output_path):
    print("Exporting to PyTorch Mobile (TorchScript)...")
    # Example input tensor
    example_input = torch.rand(1, 3, 224, 224)
    
    # Trace the model
    traced_script_module = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized_traced_model = optimize_for_mobile(traced_script_module)
        optimized_traced_model._save_for_lite_interpreter(output_path)
        print(f"✅ Saved PyTorch Mobile lite model to: {output_path}")
    except ImportError:
        # Fallback if optimize_for_mobile is not available in older torch versions
        traced_script_module.save(output_path)
        print(f"✅ Saved standard TorchScript model to: {output_path}")

def export_to_onnx(model, output_path):
    print("Exporting to ONNX format...")
    example_input = torch.rand(1, 3, 224, 224)
    
    torch.onnx.export(
        model, 
        example_input, 
        output_path, 
        export_params=True,
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Saved ONNX model to: {output_path}")

if __name__ == "__main__":
    weights_path = "best_model.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Could not find {weights_path}")
        exit(1)
        
    model = load_model(weights_path)
    
    # 1. Export to PyTorch Mobile Lite (Android/iOS)
    export_to_torchscript_mobile(model, "best_model.ptl")
    
    # 2. Export to ONNX (Universal - can be converted to CoreML / TFLite)
    export_to_onnx(model, "best_model.onnx")
