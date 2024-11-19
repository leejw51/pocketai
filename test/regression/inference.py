import torch
from model import RegressionModel

def predict(x):
    # Load the trained model
    model = RegressionModel()
    model.load_state_dict(torch.load('regression_model.pth'))
    model.eval()
    
    # Convert input to tensor
    x = torch.tensor([[x]], dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(x)
    
    return prediction.item()

def interactive_prediction():
    print("Welcome to the Regression Model Predictor!")
    print("(Enter 'q' to quit)")
    
    while True:
        try:
            user_input = input("\nEnter a number to predict its value: ")
            
            if user_input.lower() == 'q':
                print("Goodbye!")
                break
            
            x = float(user_input)
            prediction = predict(x)
            print(f"Input: {x}")
            print(f"Predicted Output: {prediction:.3f}")
            print(f"Expected (approx): {2*x + 1:.3f} (without noise)")
            
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Make sure you have trained the model first (run train.py)")
            break

if __name__ == "__main__":
    interactive_prediction() 