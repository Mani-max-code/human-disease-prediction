package com.healthpredict.service;



import com.healthpredict.model.HealthInput;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Service;
import org.springframework.core.io.ClassPathResource;

import jakarta.annotation.PostConstruct;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

@Service
public class PredictionService {

    private MultiLayerNetwork model;

    @PostConstruct
    public void loadModel() {
        try {
            File modelFile = new ClassPathResource("models/health_model.zip").getFile();
            model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            System.out.println("✅ DL4J Model loaded successfully!");
        } catch (Exception e) {
            System.out.println("❌ Model load failed: " + e.getMessage());
        }
    }

    public Map<String, Object> predict(HealthInput input) {
        Map<String, Object> result = new HashMap<>();

        try {
            // Normalize inputs manually (min-max scaling)
            double age = normalize(input.getAge(), 25, 71);
            double bmi = normalize(input.getBmi(), 18, 41);
            double bp = normalize(input.getBloodPressure(), 100, 178);
            double glucose = normalize(input.getGlucose(), 78, 255);
            double chol = normalize(input.getCholesterol(), 155, 305);
            double smoking = input.getSmoking();
            double family = input.getFamilyHistory();

            // Create input array for DL4J
            INDArray inputArray = Nd4j.create(new double[][]{
                {age, bmi, bp, glucose, chol, smoking, family}
            });

            // Get prediction from neural network
            INDArray output = model.output(inputArray);
            double diabetesProb = output.getDouble(0, 1) * 100;

            // Use scoring for hypertension and heart
            // (based on clinical weights since we trained on diabetes)
            double hypertensionRisk = calculateHypertensionRisk(input);
            double heartRisk = calculateHeartRisk(input);

            long diabetesRisk = Math.round(diabetesProb);

            result.put("diabetesRisk", diabetesRisk);
            result.put("hypertensionRisk", Math.round(hypertensionRisk));
            result.put("heartRisk", Math.round(heartRisk));
            result.put("diabetesLevel", getRiskLevel(diabetesProb));
            result.put("hypertensionLevel", getRiskLevel(hypertensionRisk));
            result.put("heartLevel", getRiskLevel(heartRisk));
            result.put("advice", getAdvice(diabetesProb, hypertensionRisk, heartRisk));

        } catch (Exception e) {
            System.out.println("Prediction error: " + e.getMessage());
            // Fallback to rule based
            double d = calculateDiabetesRisk(input);
            double h = calculateHypertensionRisk(input);
            double heart = calculateHeartRisk(input);
            result.put("diabetesRisk", Math.round(d));
            result.put("hypertensionRisk", Math.round(h));
            result.put("heartRisk", Math.round(heart));
            result.put("diabetesLevel", getRiskLevel(d));
            result.put("hypertensionLevel", getRiskLevel(h));
            result.put("heartLevel", getRiskLevel(heart));
            result.put("advice", getAdvice(d, h, heart));
        }

        return result;
    }

    private double normalize(double value, double min, double max) {
        return (value - min) / (max - min);
    }

    private double calculateDiabetesRisk(HealthInput i) {
        double risk = 0;
        if (i.getGlucose() > 140) risk += 35;
        else if (i.getGlucose() > 100) risk += 15;
        if (i.getBmi() > 30) risk += 25;
        else if (i.getBmi() > 25) risk += 10;
        if (i.getAge() > 45) risk += 15;
        if (i.getFamilyHistory() == 1) risk += 15;
        if (i.getSmoking() == 1) risk += 10;
        return Math.min(risk, 95);
    }

    private double calculateHypertensionRisk(HealthInput i) {
        double risk = 0;
        if (i.getBloodPressure() > 140) risk += 40;
        else if (i.getBloodPressure() > 120) risk += 20;
        if (i.getBmi() > 30) risk += 20;
        if (i.getAge() > 50) risk += 15;
        if (i.getSmoking() == 1) risk += 15;
        if (i.getFamilyHistory() == 1) risk += 10;
        return Math.min(risk, 95);
    }

    private double calculateHeartRisk(HealthInput i) {
        double risk = 0;
        if (i.getCholesterol() > 240) risk += 35;
        else if (i.getCholesterol() > 200) risk += 15;
        if (i.getBloodPressure() > 140) risk += 20;
        if (i.getSmoking() == 1) risk += 25;
        if (i.getAge() > 55) risk += 15;
        if (i.getFamilyHistory() == 1) risk += 15;
        return Math.min(risk, 95);
    }

    private String getRiskLevel(double risk) {
        if (risk >= 60) return "High";
        else if (risk >= 30) return "Moderate";
        else return "Low";
    }

    private String getAdvice(double d, double h, double heart) {
        if (d >= 60 || h >= 60 || heart >= 60)
            return "High risk detected. Please consult a doctor immediately.";
        else if (d >= 30 || h >= 30 || heart >= 30)
            return "Moderate risk detected. Lifestyle changes recommended.";
        else
            return "Low risk detected. Maintain your healthy lifestyle!";
    }
}
/*import com.healthpredict.model.HealthInput;
import org.springframework.stereotype.Service;
import java.util.HashMap;
import java.util.Map;

@Service
public class PredictionService {

    public Map<String, Object> predict(HealthInput input) {
        Map<String, Object> result = new HashMap<>();

        double diabetesRisk = calculateDiabetesRisk(input);
        double hypertensionRisk = calculateHypertensionRisk(input);
        double heartRisk = calculateHeartRisk(input);

        result.put("diabetesRisk", Math.round(diabetesRisk));
        result.put("hypertensionRisk", Math.round(hypertensionRisk));
        result.put("heartRisk", Math.round(heartRisk));
        result.put("diabetesLevel", getRiskLevel(diabetesRisk));
        result.put("hypertensionLevel", getRiskLevel(hypertensionRisk));
        result.put("heartLevel", getRiskLevel(heartRisk));
        result.put("advice", getAdvice(diabetesRisk, hypertensionRisk, heartRisk));

        return result;
    }

    private double calculateDiabetesRisk(HealthInput i) {
        double risk = 0;
        if (i.getGlucose() > 140) risk += 35;
        else if (i.getGlucose() > 100) risk += 15;
        if (i.getBmi() > 30) risk += 25;
        else if (i.getBmi() > 25) risk += 10;
        if (i.getAge() > 45) risk += 15;
        if (i.getFamilyHistory() == 1) risk += 15;
        if (i.getSmoking() == 1) risk += 10;
        return Math.min(risk, 95);
    }

    private double calculateHypertensionRisk(HealthInput i) {
        double risk = 0;
        if (i.getBloodPressure() > 140) risk += 40;
        else if (i.getBloodPressure() > 120) risk += 20;
        if (i.getBmi() > 30) risk += 20;
        if (i.getAge() > 50) risk += 15;
        if (i.getSmoking() == 1) risk += 15;
        if (i.getFamilyHistory() == 1) risk += 10;
        return Math.min(risk, 95);
    }

    private double calculateHeartRisk(HealthInput i) {
        double risk = 0;
        if (i.getCholesterol() > 240) risk += 35;
        else if (i.getCholesterol() > 200) risk += 15;
        if (i.getBloodPressure() > 140) risk += 20;
        if (i.getSmoking() == 1) risk += 25;
        if (i.getAge() > 55) risk += 15;
        if (i.getFamilyHistory() == 1) risk += 15;
        return Math.min(risk, 95);
    }

    private String getRiskLevel(double risk) {
        if (risk >= 60) return "High";
        else if (risk >= 30) return "Moderate";
        else return "Low";
    }

    private String getAdvice(double d, double h, double heart) {
        if (d >= 60 || h >= 60 || heart >= 60)
            return "High risk detected. Please consult a doctor immediately.";
        else if (d >= 30 || h >= 30 || heart >= 30)
            return "Moderate risk detected. Lifestyle changes recommended.";
        else
            return "Low risk detected. Maintain your healthy lifestyle!";
    }
}*/
