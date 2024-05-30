import streamlit as st
import cv2
import numpy as np
import time
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageOps

icon_path = "esophagitis_icon.ico"
st.set_page_config(page_title="Esophagitis.AI", page_icon=icon_path)

# Memuat model yang telah dilatih
model = load_model('noreso.h5')

def prediksi_gambar(file_path):
    class_names = open("noreso.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(file_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    hasil = {
        'label_kelas': class_name,
        'skor_kepercayaan': confidence_score,
    }
    return hasil

# Aplikasi Streamlit
# Navigasi
halaman_terpilih = st.selectbox("Pilih Halaman", ["Beranda", "Halaman Deteksi", "Visualisasi Model"],format_func=lambda x: x)

if halaman_terpilih == "Beranda":
    # Tampilkan Halaman Beranda

    st.header("Selamat Datang di Aplikasi Deteksi Esophagitis!", divider='rainbow')
    st.write(
        "Aplikasi ini memungkinkan Anda untuk mengunggah gambar Esophagus Anda"
        " dan mendapatkan hasil deteksi.") 
    st.write(
        "Apakah Esophagus Anda normal atau terindikasi Esophagitis?"
    )

elif halaman_terpilih == "Halaman Deteksi":
    # Tampilkan Halaman Deteksi
    st.title("Unggah Gambar")
    st.markdown("---")

    # Unggah Gambar melalui Streamlit
    berkas_gambar = st.file_uploader("Silahkan Pilih Gambar ", type=["jpg", "jpeg", "png"])
    if berkas_gambar:
        # Tampilkan gambar yang dipilih
        st.image(berkas_gambar, caption="Gambar yang Diunggah", use_column_width=True)
        if st.button("Deteksi"):
            # Simpan berkas gambar yang diunggah ke lokasi sementara
            with open("temp_image.jpg", "wb") as f :
                f.write(berkas_gambar.getbuffer())

            # Lakukan Prediksi Pada Berkas yang disimpan
            hasil_prediksi = prediksi_gambar("temp_image.jpg")
            # Tampilkan Hasil Prediksi
            st.write(f"Hasil Deteksi: {hasil_prediksi['label_kelas']}")
            st.write(f"Skor Kepercayaan: {hasil_prediksi['skor_kepercayaan']:.2%}")

            if hasil_prediksi['skor_kepercayaan'] < 0.5:
                st.write("Hasil deteksi menunjukkan bahwa gambar ini mungkin tidak sesuai dengan kategori yang dideteksi oleh model. Pastikan gambar yang diunggah adalah gambar esophagus yang benar dan coba lagi.")
            elif hasil_prediksi['label_kelas'] == 'normal' :
                st.write(
                    "Selamat! Berdasarkan Deteksi kami, Esophagus Anda tampaknya dalam keadaan normal. Namun, Ingatlah bahwa ini hanya hasil dari model kecerdasan buatan kami. Jika Anda memiliki kekhawatiran kesehatan atau pertanyaan lebih lanjut, sangat disarankan untuk berkonsultasi dengan dokter untuk pemeriksaan yang lebih mendalam.")
            elif hasil_prediksi['label_kelas'] == 'esophagitis' :
                st.write(
                    "Hasil Deteksi kami menunjukkan terdeteksi indikasi Esophagitis pada Esophagus Anda. Namun, perlu diingat bahwa ini hanya hasil dari model kecerdasan buatan kami. Kami sarankan Anda untuk segera berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut dan konfirmasi. Jangan ragu untuk mendiskusikan hasil ini bersama profesional kesehatan anda.")

else:
    st.title("Kinerja Model AI")
    st.markdown("---")

    def display_image_table(image_path1, title1, caption1, image_path2, title2, caption2):
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2) if image_path2 else None

        col1, col2 = st.columns(2)

        with col1:
            col1.markdown(f'<h2 style="text-align:center;">{title1}</h2>', unsafe_allow_html=True)
            col1.markdown(
                f'<div style="display: flex; justify-content: center;"></div>',
                unsafe_allow_html=True
            )
            col1.image(image1, use_column_width=True)
            col1.markdown(f'<p stye="text-align:left;">{caption1}</p>', unsafe_allow_html=True)
        
        with col2:
            if image2:
                col2.markdown(f'<h2 style="text-align:center;">{title2}</h2>', unsafe_allow_html=True)
            
                col2.markdown(
                    f'<div style="display: flex; justify-content: center;"></div>',
                unsafe_allow_html=True
                )
                col2.image(image2, use_column_width=True)
                col2.markdown(f'<p stye="text-align:left;">{caption2}</p>', unsafe_allow_html=True)

    images_info = [
        {'path': 'accuracy_class.png','title': 'Accuracy Class', 'caption':'Tabel ini mendeteksi esophagitis mencapai akurasi 100% untuk esophagitis dan 89% untuk kondisi normal berdasarkan evaluasi terhadap 14 sampel esophagitis dan 18 sampel normal.'},
        {'path': 'accuracy_epoch.png','title': 'Accuracy Epoch', 'caption':'Grafik menunjukkan bahwa model CNN mencapai akurasi tinggi dan stabil dengan cepat, mendekati 100% pada data pelatihan dan sekitar 90% pada data pengujian setelah beberapa epoch awal, tanpa tanda-tanda overfitting.'},
        {'path': 'matrix.png','title': 'Confusion Matrix', 'caption':'Confussion Matriks ini menunjukkan bahwa model CNN mendeteksi esophagitis dengan sempurna (14/14) dan kondisi normal dengan sedikit kesalahan (16/18), menunjukkan performa akurasi yang tinggi.'},
        {'path': 'loss_epoch.png','title': 'Loss Epoch', 'caption':'Grafik menunjukkan bahwa loss pada data pelatihan (garis biru) menurun tajam mendekati nol, sementara loss pada data pengujian (garis oranye) menurun pada awalnya tetapi kemudian stabil di sekitar 0.4, menunjukkan adanya kemungkinan overfitting setelah beberapa epoch.'},
    ]
    for i in range(0, len(images_info), 2):
        if i + 1 < len(images_info):
            display_image_table(
                images_info[i]['path'], images_info[i]['title'], images_info[i]['caption'],
                images_info[i + 1]['path'], images_info[i + 1]['title'], images_info[i + 1]['caption']
            )
        else:
            display_image_table(images_info[i]['path'], images_info[i]['title'], images_info[i]['caption'], '', '', '')
