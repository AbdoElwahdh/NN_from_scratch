package org.example.web;

import org.springframework.stereotype.Service;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Image;
import java.io.ByteArrayInputStream;
import java.io.IOException;

@Service
public class ImageProcessorService {

    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;

    public double[] processImage(byte[] imageBytes) throws IOException {
        // 1. قراءة الصورة
        ByteArrayInputStream bis = new ByteArrayInputStream(imageBytes);
        BufferedImage originalImage = ImageIO.read(bis);

        // 2. تغيير حجم الصورة إلى 28x28 وتحويلها إلى تدرج الرمادي
        Image resizedImage = originalImage.getScaledInstance(IMAGE_WIDTH, IMAGE_HEIGHT, Image.SCALE_SMOOTH);
        BufferedImage bufferedResizedImage = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        bufferedResizedImage.getGraphics().drawImage(resizedImage, 0, 0, null);

        // 3. تحويل الصورة إلى مصفوفة double[] وتطبيع القيم
        double[] imageVector = new double[IMAGE_WIDTH * IMAGE_HEIGHT];
        int k = 0;
        for (int i = 0; i < IMAGE_HEIGHT; i++) {
            for (int j = 0; j < IMAGE_WIDTH; j++) {
                // تحويل البكسل إلى قيمة بين 0 و 1
                // ملاحظة: قد تحتاج لعكس الألوان (255 - pixel) إذا كان النموذج مدربًا على خلفية سوداء
                int pixel = bufferedResizedImage.getRGB(j, i) & 0xFF;
                imageVector[k++] = pixel / 255.0;
            }
        }
        return imageVector;
    }
}
