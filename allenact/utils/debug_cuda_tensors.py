from typing import Any, Dict, List, Set
import inspect

import torch


def find_cuda_tensors(obj: Any, path: str = "root", visited: Set[int] = None, max_depth: int = 10) -> List[str]:
    """
    Recursively search for CUDA tensors in an object and return their paths.
    
    Args:
        obj: Object to search
        path: Current path in the object hierarchy
        visited: Set of object IDs we've already visited (to avoid cycles)
        max_depth: Maximum recursion depth
    
    Returns:
        List of paths where CUDA tensors were found
    """
    if visited is None:
        visited = set()
    
    cuda_paths = []
    
    # Avoid infinite recursion
    if max_depth <= 0:
        return cuda_paths
    
    # Avoid cycles
    obj_id = id(obj)
    if obj_id in visited:
        return cuda_paths
    visited.add(obj_id)
    
    try:
        # Check if this object is a CUDA tensor
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            cuda_paths.append(f"{path} (shape: {obj.shape}, device: {obj.device}, dtype: {obj.dtype})")
            return cuda_paths
        
        # Check if this object has a device attribute that points to CUDA
        if hasattr(obj, 'device') and hasattr(obj.device, 'type') and obj.device.type == 'cuda':
            cuda_paths.append(f"{path}.device = {obj.device}")
        
        # Recursively check different types of containers
        if isinstance(obj, dict):
            for key, value in obj.items():
                try:
                    cuda_paths.extend(find_cuda_tensors(
                        value, f"{path}['{key}']", visited.copy(), max_depth - 1
                    ))
                except Exception as e:
                    cuda_paths.append(f"{path}['{key}'] - Error accessing: {str(e)}")
        
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                try:
                    cuda_paths.extend(find_cuda_tensors(
                        item, f"{path}[{i}]", visited.copy(), max_depth - 1
                    ))
                except Exception as e:
                    cuda_paths.append(f"{path}[{i}] - Error accessing: {str(e)}")
        
        elif hasattr(obj, '__dict__'):
            # Check object attributes
            for attr_name in dir(obj):
                # Skip private/special methods and properties that might cause issues
                if (attr_name.startswith('_') or 
                    attr_name in ['__class__', '__dict__', '__weakref__'] or
                    callable(getattr(obj, attr_name, None))):
                    continue
                
                try:
                    attr_value = getattr(obj, attr_name)
                    cuda_paths.extend(find_cuda_tensors(
                        attr_value, f"{path}.{attr_name}", visited.copy(), max_depth - 1
                    ))
                except (AttributeError, RuntimeError, TypeError) as e:
                    # Some attributes might not be accessible or might be properties
                    # that cause issues when accessed
                    continue
                except Exception as e:
                    cuda_paths.append(f"{path}.{attr_name} - Error accessing: {str(e)}")
    
    except Exception as e:
        cuda_paths.append(f"{path} - Error processing object: {str(e)}")
    
    return cuda_paths


def check_sampler_args_for_cuda(sampler_args: Dict[str, Any]) -> None:
    """
    Check task sampler arguments for CUDA tensors and print results.
    
    Args:
        sampler_args: Dictionary of arguments passed to task sampler
    """
    print("🔍 Checking task sampler arguments for CUDA tensors...")
    print("=" * 60)
    
    # Check current CUDA memory usage
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated()
        print(f"📊 Current CUDA memory allocated: {current_memory / (1024**2):.2f} MB")
        print()
    
    cuda_tensors = find_cuda_tensors(sampler_args, "sampler_args")
    
    if cuda_tensors:
        print("🚨 FOUND CUDA TENSORS:")
        for i, path in enumerate(cuda_tensors, 1):
            print(f"  {i}. {path}")
    else:
        print("✅ No CUDA tensors found in sampler arguments")
    
    print("=" * 60)


def check_specific_objects_for_cuda(sampler_args: Dict[str, Any]) -> None:
    """
    Check specific high-priority objects that are most likely to contain CUDA tensors.
    """
    print("🎯 Checking specific objects for CUDA tensors...")
    print("=" * 60)
    
    high_priority_checks = [
        ('sensors', sampler_args.get('task_args', {}).get('sensors', [])),
        ('action_space', sampler_args.get('task_args', {}).get('action_space')),
        ('controller_args', sampler_args.get('controller_args', {})),
        ('action_hook_runner', sampler_args.get('controller_args', {}).get('action_hook_runner')),
    ]
    
    for name, obj in high_priority_checks:
        if obj is not None:
            print(f"\n🔍 Checking {name}:")
            cuda_tensors = find_cuda_tensors(obj, name, max_depth=5)
            if cuda_tensors:
                print(f"  🚨 Found CUDA tensors in {name}:")
                for path in cuda_tensors:
                    print(f"    - {path}")
            else:
                print(f"  ✅ No CUDA tensors in {name}")
    
    print("=" * 60)


def debug_cuda_in_task_sampler():
    """
    Add this function call to your task sampler creation to debug CUDA usage.
    """
    import sys
    import traceback
    
    # Get the calling frame to access local variables
    frame = sys._getframe(1)
    locals_dict = frame.f_locals

    # Look for sampler arguments in the calling scope
    potential_args = [
        'kwargs', 'sampler_args', 'args', 'task_sampler_args', 'current_sampler_fn_args_list', 'callback_sensor_suite'
    ]
    
    for arg_name in potential_args:
        if arg_name in locals_dict:
            args = locals_dict[arg_name]
            if isinstance(args, dict):
                print(f"\n🔍 Found {arg_name} in calling scope, checking for CUDA tensors...")
                check_sampler_args_for_cuda(args)
                check_specific_objects_for_cuda(args)
                break
    else:
        print("⚠️  Could not find sampler arguments in calling scope")
        print("Available variables:", list(locals_dict.keys()))


if __name__ == "__main__":
    # Example usage
    sampler_args = {
        'device': 1,
        'sensors': [],
        'some_tensor': torch.randn(3, 3).cuda() if torch.cuda.is_available() else torch.randn(3, 3),
        'nested': {
            'data': torch.zeros(10).cuda() if torch.cuda.is_available() else torch.zeros(10)
        }
    }
    
    debug_cuda_in_task_sampler()
