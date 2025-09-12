package org.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class WebApp {
    public static void main(String[] args) {
        // هذا السطر سيقوم بتشغيل خادم الويب
        SpringApplication.run(WebApp.class, args);
    }
}
