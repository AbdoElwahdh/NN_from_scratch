package org.example.web;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@RestController
public class PredictionController {

    @Autowired
    private NeuralNetworkService neuralNetworkService;

    @Autowired
    private ImageProcessorService imageProcessorService;

    @PostMapping("/predict" )
    public ResponseEntity<Map<String, Object>> predict(@RequestParam("image") MultipartFile imageFile) {
        try {
            if (imageFile.isEmpty()) {
                return ResponseEntity.badRequest().body(Map.of("error", "Image file is empty"));
            }

            // 1. تحويل الصورة إلى مصفوفة يقبلها النموذج
            double[] input = imageProcessorService.processImage(imageFile.getBytes());

            // 2. استدعاء النموذج للتنبؤ
            int prediction = neuralNetworkService.predict(input);

            Map<String, Object> response = Map.of("prediction", prediction);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.internalServerError().body(Map.of("error", "Failed to process image: " + e.getMessage()));
        }
    }
}
