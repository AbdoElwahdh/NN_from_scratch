// انتظر حتى يتم تحميل الصفحة بالكامل
document.addEventListener('DOMContentLoaded', () => {

    // الحصول على العناصر من الصفحة
    const imageInput = document.getElementById('image-file');
    const resultBox = document.getElementById('result-box');
    const preview = document.getElementById('preview');

    // إضافة مستمع لحدث تغيير حقل إدخال الصورة
    imageInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) {
            return; // إذا لم يختر المستخدم ملفًا، لا تفعل شيئًا
        }

        // إظهار الصورة المصغرة (preview)
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.innerHTML = `<img src="${e.target.result}" alt="Image preview"/>`;
        };
        reader.readAsDataURL(file);

        // إرسال النموذج تلقائيًا بمجرد اختيار الصورة
        submitForm(file);
    });

    // دالة لإرسال الصورة إلى الخادم
    async function submitForm(file) {
        const formData = new FormData();
        formData.append('image', file);

        resultBox.textContent = '...'; // إظهار علامة التحميل

        try {
            // إرسال طلب POST إلى الخادم
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            // قراءة الرد كـ JSON
            const data = await response.json();

            if (response.ok) {
                // إذا نجح الطلب، اعرض النتيجة
                resultBox.textContent = `${data.prediction}`;
            } else {
                // إذا فشل الطلب، اعرض رسالة خطأ
                resultBox.textContent = `Error`;
                alert(data.error); // إظهار تفاصيل الخطأ في نافذة منبثقة
            }
        } catch (error) {
            // في حالة حدوث خطأ في الشبكة أو الاتصال
            resultBox.textContent = 'Error';
            alert('An unexpected network error occurred. Check the console.');
            console.error('Error:', error);
        }
    }
});
