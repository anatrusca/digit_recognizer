import subprocess

print("Preprocessing dataset...")
subprocess.run(["python", "-m", "utils.preprocessing"])

print("Training model with cross-validation and early stopping...")
subprocess.run(["python", "-m", "model.train"])

print("Evaluating model...")
subprocess.run(["python", "-m", "model.evaluate"])

print("All steps completed successfully!")
