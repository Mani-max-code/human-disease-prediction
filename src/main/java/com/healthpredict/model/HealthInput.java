package com.healthpredict.model;

public class HealthInput {
    private double age;
    private double bmi;
    private double bloodPressure;
    private double systolicBP;
    private double diastolicBP;
    private double glucose;
    private double cholesterol;
    private double smoking;      // 1 = yes, 0 = no
    private double familyHistory; // 1 = yes, 0 = no

    // Getters and Setters
    public double getAge() { return age; }
    public void setAge(double age) { this.age = age; }

    public double getBmi() { return bmi; }
    public void setBmi(double bmi) { this.bmi = bmi; }

    public double getSystolicBP() { return systolicBP; }
    public void setSystolicBP(double systolicBP) { this.systolicBP = systolicBP; }

    public double getDiastolicBP() { return diastolicBP; }
    public void setDiastolicBP(double diastolicBP) { this.diastolicBP = diastolicBP; }
    public double getGlucose() { return glucose; }
    public void setGlucose(double glucose) { this.glucose = glucose; }

    public double getCholesterol() { return cholesterol; }
    public void setCholesterol(double cholesterol) { this.cholesterol = cholesterol; }

    public double getSmoking() { return smoking; }
    public void setSmoking(double smoking) { this.smoking = smoking; }

    public double getFamilyHistory() { return familyHistory; }
    public void setFamilyHistory(double familyHistory) { this.familyHistory = familyHistory; }
}
