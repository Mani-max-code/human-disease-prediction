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

    private MultiLayerNetwork diabetesModel;
    private MultiLayerNetwork hypertensionModel;
    private MultiLayerNetwork heartModel;

    @PostConstruct
    public void loadModels() {
        try {
            File diabetesFile = new ClassPathResource(
                "models/Diabetes_model.zip").getFile();
            diabetesModel = ModelSerializer
                .restoreMultiLayerNetwork(diabetesFile);
            System.out.println("✅ Diabetes model loaded!");

            File hypertensionFile = new ClassPathResource(
                "models/Hypertension_model.zip").getFile();
            hypertensionModel = ModelSerializer
                .restoreMultiLayerNetwork(hypertensionFile);
            System.out.println("✅ Hypertension model loaded!");

            File heartFile = new ClassPathResource(
                "models/Heart_model.zip").getFile();
            heartModel = ModelSerializer
                .restoreMultiLayerNetwork(heartFile);
            System.out.println("✅ Heart model loaded!");

        } catch (Exception e) {
            System.out.println("❌ Model load failed: " + e.getMessage());
        }
    }

    public Map<String, Object> predict(HealthInput input) {
        Map<String, Object> result = new HashMap<>();

        // Check crisis first
        boolean isCrisis = input.getSystolicBP() >= 180 ||
                           input.getDiastolicBP() >= 120;

        try {
            // Normalize inputs
            double age    = normalize(input.getAge(), 25, 71);
            double bmi    = normalize(input.getBmi(), 18, 41);
            double sys    = normalize(input.getSystolicBP(), 90, 200);
            double dia    = normalize(input.getDiastolicBP(), 40, 130);
            double gluc   = normalize(input.getGlucose(), 78, 255);
            double chol   = normalize(input.getCholesterol(), 155, 305);
            double smoke  = input.getSmoking();
            double family = input.getFamilyHistory();

            // Create input array
            INDArray inputArray = Nd4j.create(new double[][]{
                {age, bmi, sys, dia, gluc, chol, smoke, family}
            });

            // Step 1 — Get BASE probability from each DL4J model
            double diabetesBase = diabetesModel
                .output(inputArray).getDouble(0, 1) * 100;
            double hypertensionBase = hypertensionModel
                .output(inputArray).getDouble(0, 1) * 100;
            double heartBase = heartModel
                .output(inputArray).getDouble(0, 1) * 100;

            System.out.println("=== DL4J Base Scores ===");
            System.out.println("Diabetes base:     " + diabetesBase);
            System.out.println("Hypertension base: " + hypertensionBase);
            System.out.println("Heart base:        " + heartBase);

            // Step 2 — Apply clinical adjustments per disease
            double diabetesFinal     = adjustDiabetes(diabetesBase, input);
            double hypertensionFinal = adjustHypertension(hypertensionBase, input);
            double heartFinal        = adjustHeart(heartBase, input);

            System.out.println("=== Final Scores After Adjustment ===");
            System.out.println("Diabetes final:     " + diabetesFinal);
            System.out.println("Hypertension final: " + hypertensionFinal);
            System.out.println("Heart final:        " + heartFinal);

            result.put("diabetesRisk",
                Math.min(Math.round(diabetesFinal), 99));
            result.put("hypertensionRisk",
                Math.min(Math.round(hypertensionFinal), 99));
            result.put("heartRisk",
                Math.min(Math.round(heartFinal), 99));
            result.put("diabetesLevel",
                getRiskLevel(diabetesFinal));
            result.put("hypertensionLevel",
                isCrisis ? "Crisis" : getRiskLevel(hypertensionFinal));
            result.put("heartLevel",
                getRiskLevel(heartFinal));
            result.put("isCrisis", isCrisis);
            result.put("advice",
                getAdvice(diabetesFinal,
                          hypertensionFinal,
                          heartFinal, isCrisis));

        } catch (Exception e) {
            System.out.println("Prediction error: " + e.getMessage());
            e.printStackTrace();
        }

        return result;
    }

    // ── DIABETES ADJUSTMENT ──────────────────────────────
    // Key factors: Glucose, BMI, Age, Family History
    private double adjustDiabetes(double base, HealthInput i) {
        double adjustment = 0;

        // Glucose is strongest diabetes indicator
        if (i.getGlucose() >= 200)      adjustment += 30;
        else if (i.getGlucose() >= 126) adjustment += 20;
        else if (i.getGlucose() >= 100) adjustment += 5;
        else if (i.getGlucose() < 100)  adjustment -= 15; // Normal glucose reduces risk

        // BMI contribution
        if (i.getBmi() >= 35)           adjustment += 15;
        else if (i.getBmi() >= 30)      adjustment += 10;
        else if (i.getBmi() >= 25)      adjustment += 3;
        else if (i.getBmi() < 25)       adjustment -= 5; // Normal BMI reduces risk

        // Age
        if (i.getAge() >= 65)           adjustment += 8;
        else if (i.getAge() >= 45)      adjustment += 4;
        else if (i.getAge() < 35)       adjustment -= 5; // Young age reduces risk

        // Family history
        if (i.getFamilyHistory() == 1)  adjustment += 8;
        else                            adjustment -= 3;

        // Smoking
        if (i.getSmoking() == 1)        adjustment += 4;

        // Combine: 60% DL4J + 40% clinical adjustment
        double final_score = (base * 0.6) + ((base + adjustment) * 0.4);
        return Math.max(0, Math.min(final_score, 99));
    }

    // ── HYPERTENSION ADJUSTMENT ──────────────────────────
    // Key factors: Systolic BP, Diastolic BP, BMI, Age
    private double adjustHypertension(double base, HealthInput i) {
        double adjustment = 0;

        // Systolic BP is strongest hypertension indicator
        if (i.getSystolicBP() >= 180)      adjustment += 40;
        else if (i.getSystolicBP() >= 160) adjustment += 28;
        else if (i.getSystolicBP() >= 140) adjustment += 18;
        else if (i.getSystolicBP() >= 130) adjustment += 8;
        else if (i.getSystolicBP() >= 120) adjustment += 2;
        else                               adjustment -= 15; // Normal BP reduces risk

        // Diastolic BP
        if (i.getDiastolicBP() >= 120)     adjustment += 20;
        else if (i.getDiastolicBP() >= 110) adjustment += 15;
        else if (i.getDiastolicBP() >= 90) adjustment += 10;
        else if (i.getDiastolicBP() >= 80) adjustment += 3;
        else                               adjustment -= 8; // Normal reduces risk

        // BMI
        if (i.getBmi() >= 30)              adjustment += 8;
        else if (i.getBmi() < 25)          adjustment -= 5;

        // Age
        if (i.getAge() >= 60)              adjustment += 8;
        else if (i.getAge() >= 50)         adjustment += 4;
        else if (i.getAge() < 35)          adjustment -= 5;

        // Smoking
        if (i.getSmoking() == 1)           adjustment += 6;

        // Family history
        if (i.getFamilyHistory() == 1)     adjustment += 5;
        else                               adjustment -= 3;

        // Combine: 50% DL4J + 50% clinical adjustment
        double final_score = (base * 0.5) + ((base + adjustment) * 0.5);
        return Math.max(0, Math.min(final_score, 99));
    }

    // ── HEART DISEASE ADJUSTMENT ─────────────────────────
    // Key factors: Cholesterol, BP, Smoking, Age
    private double adjustHeart(double base, HealthInput i) {
        double adjustment = 0;

        // Cholesterol is strongest heart disease indicator
        if (i.getCholesterol() >= 280)      adjustment += 30;
        else if (i.getCholesterol() >= 240) adjustment += 20;
        else if (i.getCholesterol() >= 200) adjustment += 8;
        else                                adjustment -= 15; // Normal reduces risk

        // Blood pressure contribution to heart
        if (i.getSystolicBP() >= 160)       adjustment += 18;
        else if (i.getSystolicBP() >= 140)  adjustment += 10;
        else if (i.getSystolicBP() < 120)   adjustment -= 8;

        // Smoking — very strong heart risk
        if (i.getSmoking() == 1)            adjustment += 18;
        else                                adjustment -= 5;

        // Age
        if (i.getAge() >= 65)               adjustment += 12;
        else if (i.getAge() >= 55)          adjustment += 8;
        else if (i.getAge() >= 45)          adjustment += 4;
        else if (i.getAge() < 40)           adjustment -= 8;

        // Family history
        if (i.getFamilyHistory() == 1)      adjustment += 8;
        else                                adjustment -= 3;

        // Combine: 50% DL4J + 50% clinical adjustment
        double final_score = (base * 0.5) + ((base + adjustment) * 0.5);
        return Math.max(0, Math.min(final_score, 99));
    }

    private double normalize(double value, double min, double max) {
        return (value - min) / (max - min);
    }

    private String getRiskLevel(double risk) {
        if (risk >= 60)      return "High";
        else if (risk >= 30) return "Moderate";
        else                 return "Low";
    }

    private String getAdvice(double d, double h,
                             double heart, boolean crisis) {
        if (crisis)
            return "🚨 HYPERTENSIVE CRISIS! Seek emergency medical care immediately!";
        else if (d >= 60 || h >= 60 || heart >= 60)
            return "High risk detected. Please consult a doctor immediately.";
        else if (d >= 30 || h >= 30 || heart >= 30)
            return "Moderate risk detected. Lifestyle changes recommended.";
        else
            return "Low risk detected. Maintain your healthy lifestyle!";
    }
}
/*
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
}    */
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
