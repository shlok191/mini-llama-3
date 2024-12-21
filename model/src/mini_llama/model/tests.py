import torch
import torch.nn as nn
from mini_llama.model import MiniLlamaForCausalLM

def get_default_config():
    """
    Provides a standard configuration for testing the model.
    These parameters represent a smaller version of the model
    that's suitable for testing while maintaining the essential architecture.
    """
    return {
        'vocab_size': 8192,
        'embedding_dim': 1024,
        'num_decoder_layers': 4,
        'num_attn_heads': 4,
        'mlp_layer_intermediate_dim': 2048,
        'dropout': 0.0,
        'padding_idx': 0
    }

def test_model():
    """
    Comprehensive test suite for the MiniLlamaForCausalLM model.
    Tests forward pass shapes, gradient flow, and loss computation.
    """
    
    print("\n=== Testing MiniLlama Model ===")
    
    def test_forward_shape(model, config):
        """Tests if the model produces outputs of expected shapes"""
        
        print("\nTesting forward pass shapes...")
        
        # Create input sequence
        seq_length = 512
        
        input_ids = torch.randint(0, config['vocab_size'], 
                                (seq_length,), 
                                dtype=torch.int32,
                                device='cuda')
        
        # Perform forward pass without gradients for efficiency
        with torch.no_grad():
            logits = model(input_ids, labels=None)
        
        # Check output shape
        expected_shape = (seq_length, config['vocab_size'])
        
        if logits.shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {logits.shape}")
        
        print("✓ Forward pass shape test passed")
        return model  # Return model for use in other tests
    
    def test_backward_pass(model):
        """Tests gradient computation and parameter updates"""
        
        print("\nTesting backward pass and gradient flow...")
        
        # Prepare inputs
        seq_length = 512
        config = get_default_config()
        
        input_ids = torch.randint(0, config['vocab_size'], 
                        (seq_length,),
                        dtype=torch.int32,
                        device='cuda')
        
        labels = torch.randint(0, config['vocab_size'], 
                        (seq_length,),
                        dtype=torch.int32,
                        device='cuda')
        
        # Storing the initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Computing the loss and backpropagating it :)
        loss = model(input_ids, labels)
        
        print(f"Initial loss value: {loss.item():.4f}")
        loss.backward()
        
        # Check gradients
        gradient_problems = []
        
        for name, param in model.named_parameters():
        
            if param.grad is None:
                gradient_problems.append(f"No gradient for {name}")
        
            elif torch.allclose(param.grad, torch.zeros_like(param.grad)):
                gradient_problems.append(f"Zero gradient for {name}")
        
        # Print any gradient problems that might have arised!
        if gradient_problems:
            
            raise ValueError("\n".join(gradient_problems))
        
        # Updating our  parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.step()
        
        # Verify parameter updates
        update_problems = []
        
        for name, param in model.named_parameters():
        
            if torch.allclose(param, initial_params[name]):
                update_problems.append(f"Parameter {name} didn't update after optimization")
        
        if update_problems:
            raise ValueError("\n".join(update_problems))
        
        print("✓ Backward pass and gradient flow test passed")
    
    def test_save_and_load(model, config):
        """Tests if the model can be saved and loaded correctly while maintaining its state"""
    
        print("\nTesting model saving and loading...")
        
        import os
        
        # Create a test input to compare outputs
        seq_length = 512
        test_input = torch.randint(0, config['vocab_size'], 
                                (seq_length,), 
                                dtype=torch.int32,
                                device='cuda')
        
        # Get output from original model
        with torch.no_grad():
            original_output = model(test_input, labels=None)
        
        # Save the model
        save_path = "test_model_save.pt"
        torch.save(model.state_dict(), save_path)
        
        # Create a new model instance and load the saved weights
        loaded_model = MiniLlamaForCausalLM(**config)
        loaded_model.load_state_dict(torch.load(save_path, weights_only=True))
        
        loaded_model = loaded_model.cuda() 
        
        # Get output from loaded model
        with torch.no_grad():
            loaded_output = loaded_model(test_input, labels=None)
        
        # Compare outputs
        if not torch.allclose(original_output, loaded_output, atol=1e-5):
            raise ValueError("Loaded model produces different outputs")
        
        # Clean up the saved file
        os.remove(save_path)
    
        print("✓ Model save/load test passed")


    try:
        
        # Initialize model with default configuration
        config = get_default_config()
        model = MiniLlamaForCausalLM(**config).cuda()
        
        print("✓ Model initialized successfully")
        
        # Run all tests in sequence
        test_forward_shape(model, config)
        test_backward_pass(model)
        test_save_and_load(model, config)
        
        print("\n=== All model tests passed successfully! ===")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_model()