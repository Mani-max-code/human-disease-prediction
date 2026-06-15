
# Human Disease Prediction System

A web application that predicts health risks for Diabetes, Hypertension and Heart Disease using Deep Learning.

## Project Structure

HumanDisease-Predction/
├── src/main/java/com/healthpredict/
│   ├── ModelTrainer.java
│   ├── controller/
│   │   └── PredictController.java
│   ├── model/
│   │   └── HealthInput.java
│   └── service/
│       └── PredictionService.java
├── src/main/resources/
│   ├── data/
│   │   ├── diabetes_data.csv
│   │   ├── hypertension_data.csv
│   │   └── heart_data.csv
│   ├── models/
│   │   ├── Diabetes_model.zip
│   │   ├── Hypertension_model.zip
│   │   └── Heart_model.zip
│   └── templates/
│       ├── index.html
│       └── result.html
└── pom.xml

## Tech Stack
- Java, Spring Boot, Spring MVC
- Deeplearning4j (DL4J)
- HTML, CSS, JavaScript, Thymeleaf
- Maven, Git

## How It Works
User enters 8 health parameters → 3 DL4J Neural Networks analyze input → Risk percentage shown for each disease

## Deep Learning Architecture
- 3 separate Neural Networks trained using Deeplearning4j
- Each model: Input(8) → Dense(16) → Dense(8) → Output(2)
- Activation: ReLU hidden layers, Softmax output
- Optimizer: Adam learning rate 0.001
- Training: 200 epochs
- Loss reduced from 0.66 to 0.15 during training

## Input Parameters
- Age
- BMI
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Glucose Level
- Cholesterol
- Smoking Status
- Family History

## Predictions
- Diabetes Risk
- Hypertension Risk
- Heart Disease Risk
- Hypertensive Crisis Alert when Systolic is 180 or above

## Features
- Real-time BMI Calculator
- Color coded risk cards Low Moderate High Crisis
- Progress bars for each disease
- Printable health report
- Emergency alert for hypertensive crisis

## How To Run
- Install Java 17 and Maven
- Clone the repository
- Run mvn spring-boot:run
- Open browser at http://localhost:8080

## Future Improvements
- Train on larger datasets like PIMA Indians Diabetes Database
- Add patient history tracking
- Export report as PDF
- Deploy on cloud

## Disclaimer
This application is for educational purposes only.
Always consult a qualified medical professional.

## Developer
Manishankar
GitHub: https://github.com/Mani-max-code


