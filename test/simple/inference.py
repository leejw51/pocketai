import torch
import math
from model import SineNet, degrees_to_radians

def load_model():
    model = SineNet()
    model.load_state_dict(torch.load('sine_model.pth'))
    model.eval()
    return model

def predict_sine(model, degrees):
    with torch.no_grad():
        x = torch.FloatTensor([[degrees]])
        predicted = model(x / 360.0)
        return predicted.item()

def main():
    model = load_model() 
    
    while True:
        try:
            degrees = float(input("Enter angle in degrees (or 'q' to quit): "))
            predicted_sine = predict_sine(model, degrees)
            actual_sine = math.sin(degrees_to_radians(degrees))
            
            print(f"\nAngle: {degrees}°")
            print(f"Predicted sine: {predicted_sine:.4f}")
            print(f"Actual sine: {actual_sine:.4f}")
            print(f"Difference: {abs(predicted_sine - actual_sine):.4f}")
            
        except ValueError:
            print("Goodbye!")
            break

if __name__ == "__main__":
    main() 