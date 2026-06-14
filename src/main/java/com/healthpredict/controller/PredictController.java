package com.healthpredict.controller;

import com.healthpredict.model.HealthInput;
import com.healthpredict.service.PredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;

@Controller
public class PredictController {

    @Autowired
    private PredictionService predictionService;

    @GetMapping("/")
    public String home(Model model) {
        model.addAttribute("healthInput", new HealthInput());
        return "index";
    }

    @PostMapping("/predict")
    public String predict(
        @ModelAttribute HealthInput healthInput,
        Model model) {

        var result = predictionService.predict(healthInput);

        model.addAttribute("diabetesRisk", result.get("diabetesRisk"));
        model.addAttribute("hypertensionRisk", result.get("hypertensionRisk"));
        model.addAttribute("heartRisk", result.get("heartRisk"));
        model.addAttribute("diabetesLevel", result.get("diabetesLevel"));
        model.addAttribute("hypertensionLevel", result.get("hypertensionLevel"));
        model.addAttribute("heartLevel", result.get("heartLevel"));
        model.addAttribute("advice", result.get("advice"));
        model.addAttribute("healthInput", healthInput);

        return "result";
    }
}