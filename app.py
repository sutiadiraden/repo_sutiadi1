import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
from ultralytics import YOLO

# ------------ Konfigurasi Awal -------------
st.set_page_config(
    page_title="Aplikasi Deteksi Benthic dengan YOLO (Frame First)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Aplikasi
st.title("Aplikasi Deteksi Benthic Coral - Frame Crop First")

st.markdown("""
Aplikasi ini **pertama** akan mendeteksi bounding box untuk kelas `Frame`,  
kemudian memotong (crop) gambar berdasarkan bounding box `Frame` terbesar,  
lalu **baru** menjalankan deteksi benthic pada gambar hasil crop.  

Ketika tombol **Reset Perhitungan** ditekan, seluruh data perhitungan dan file uploader akan **benar-benar direset**, 
sehingga file yang telah diunggah sebelumnya tidak muncul lagi.
""")

# ============================================
# SIDEBAR: Konfigurasi Model & Parameter Frame
# ============================================
st.sidebar.title("Konfigurasi Model")
uploaded_model = st.sidebar.file_uploader("Upload File Model YOLO (best.pt)", type=["pt"])
model_path_temp = None

if uploaded_model is not None:
    with open("temp_model.pt", "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path_temp = "temp_model.pt"

if st.sidebar.button("Load Model"):
    if model_path_temp is None:
        st.sidebar.error("Harap upload file model YOLO terlebih dahulu.")
    else:
        st.session_state['model'] = YOLO(model_path_temp)
        st.sidebar.success("Model berhasil dimuat!")

# Cek apakah model sudah dimuat
model_loaded = 'model' in st.session_state

# Parameter Frame
st.sidebar.title("Parameter Frame")
frame_width_cm = st.sidebar.number_input("Lebar Frame (cm)", value=58)
frame_height_cm = st.sidebar.number_input("Tinggi Frame (cm)", value=44)

# ============================================
# RESET COUNTER DAN PIXEL COUNTS
# ============================================
# Gunakan counter untuk memicu pergantian key di file uploader.
if 'reset_counter' not in st.session_state:
    st.session_state['reset_counter'] = 0

if 'pixel_counts' not in st.session_state:
    st.session_state['pixel_counts'] = {}

if st.sidebar.button("Reset Perhitungan"):
    # Kosongkan data perhitungan
    st.session_state['pixel_counts'] = {}
    # Tambahkan nilai counter agar key di file_uploader berubah
    st.session_state['reset_counter'] += 1
    # Rerun agar perubahan key diaplikasikan
    st.experimental_rerun()

# ============================================
# BAGIAN UTAMA
# ============================================

# Key dinamis untuk file uploader
uploader_key = f"file_uploader_{st.session_state['reset_counter']}"

st.subheader("1) Upload Gambar Benthic")
uploaded_images = st.file_uploader(
    "Pilih satu atau beberapa gambar benthic (JPG, PNG, BMP, TIFF)",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
    key=uploader_key
)

# Daftar kelas bentik
class_names = [
    "AA", "AC", "CA", "DC", "DCA", "HA", "MA", "NAC", "OT",
    "R", "RK", "S", "SC", "SI", "SP", "TA", "ZO"
]

# Kelas "Frame"
frame_class_name = "Frame"

# Inisialisasi pixel_counts jika masih kosong
if len(st.session_state['pixel_counts']) == 0:
    st.session_state['pixel_counts'] = {name: 0 for name in class_names}

# Tombol proses
if st.button("2) Jalankan Deteksi & Perhitungan (Frame -> Crop -> Benthic)"):
    if not model_loaded:
        st.warning("Harap Load Model YOLO terlebih dahulu di sidebar!")
    else:
        if not uploaded_images:
            st.warning("Harap upload minimal satu gambar terlebih dahulu.")
        else:
            st.success("Memulai proses deteksi...")

            for uploaded_file in uploaded_images:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    st.error(f"Gambar '{uploaded_file.name}' tidak dapat dibaca.")
                    continue

                # Konversi BGR -> RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 1) Deteksi FRAME
                frame_results = st.session_state['model'].predict(image_rgb, verbose=False)
                largest_area = -1
                frame_bbox = None

                # Cari bounding box Frame terbesar
                for result in frame_results:
                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        if class_name == frame_class_name:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            area = (x2 - x1) * (y2 - y1)
                            if area > largest_area:
                                largest_area = area
                                frame_bbox = (x1, y1, x2, y2)

                # 2) Crop gambar berdasarkan bounding box Frame
                if frame_bbox is not None and largest_area > 0:
                    x1, y1, x2, y2 = frame_bbox
                    cropped_img = image_rgb[y1:y2, x1:x2]
                    st.write(f"**Frame terbesar** di '{uploaded_file.name}' -> area: {largest_area}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image_rgb, caption="Gambar Asli", use_column_width=True)
                    with col2:
                        st.image(cropped_img, caption="Hasil Crop (Frame)", use_column_width=True)

                    # 3) Deteksi benthic pada gambar crop
                    benthic_results = st.session_state['model'].predict(cropped_img, verbose=False)
                    annotated_cropped = benthic_results[0].plot()
                    st.image(
                        annotated_cropped,
                        caption="Deteksi Benthic pada Gambar Crop",
                        use_column_width=True
                    )

                    # 4) Hitung piksel bounding box setiap kelas (selain Frame)
                    for result in benthic_results:
                        for i, box in enumerate(result.boxes):
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]

                            # Abaikan bounding box 'Frame' pada gambar crop
                            if class_name == frame_class_name:
                                continue

                            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
                            crop_box_area_pixel = (cx2 - cx1) * (cy2 - cy1)

                            # Akumulasi ke dictionary
                            if class_name in st.session_state['pixel_counts']:
                                st.session_state['pixel_counts'][class_name] += crop_box_area_pixel
                            else:
                                st.session_state['pixel_counts'][class_name] = crop_box_area_pixel

                else:
                    st.warning(f"Tidak ditemukan bounding box 'Frame' pada '{uploaded_file.name}'. Deteksi benthic tidak dijalankan.")

            st.success("Proses deteksi frame & benthic selesai! Data piksel diperbarui.")

# ------------------------------------------
# Bagian Perhitungan dan Grafik
# ------------------------------------------
st.subheader("3) Hasil Perhitungan Tutupan Benthic (Pasca-Crop)")

pixel_counts = st.session_state['pixel_counts']
total_pixels = sum(pixel_counts.values())

# Luas total frame (cm²)
frame_area_cm2 = frame_width_cm * frame_height_cm

# Kalkulasi persentase
if total_pixels > 0:
    class_percentages = {k: (v / total_pixels * 100) for k, v in pixel_counts.items()}
else:
    class_percentages = {k: 0 for k in pixel_counts}

df = pd.DataFrame({
    'Kelas': list(class_percentages.keys()),
    'Persentase (%)': list(class_percentages.values())
})
df = df.sort_values(by='Persentase (%)', ascending=False)

st.dataframe(df, use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df['Kelas'], df['Persentase (%)'])
ax.set_title("Persentase Tutupan Berdasarkan Kelas (Pasca-Crop)")
ax.set_xlabel("Kelas")
ax.set_ylabel("Persentase (%)")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# ------------------------------------------
# Bagian Download (chart) dan CSV
# ------------------------------------------
st.subheader("4) Unduh Hasil")

img_buffer = BytesIO()
fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
img_buffer.seek(0)
st.download_button(
    label="Unduh Grafik (PNG)",
    data=img_buffer,
    file_name="benthic_chart.png",
    mime="image/png"
)

csv_buffer = BytesIO()
df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)
st.download_button(
    label="Unduh Data CSV",
    data=csv_buffer,
    file_name="benthic_result.csv",
    mime="text/csv"
)

# Informasi Tambahan
st.info("""
**Alur Kerja**  
1. Aplikasi akan mendeteksi **Frame** pada gambar yang diunggah.  
2. Jika aplikasi menemukan **Frame** , maka gambar akan di-*crop* mengikuti ukuran frame besi tersebut.  
3. Selanjutnya, jalankan deteksi, pendeteksian benthic pada gambar hasil *crop* dan menghitung piksel masker segmentasi untuk setiap kelas benthic.  
4. Aplikasi menampilkan persentase tutupan serta menyediakan tombol unduh CSV/PNG.  
5. Apabila menekan **Reset Perhitungan**, aplikasi menghapus data perhitungan dan mereset *file uploader* sehingga file yang pernah diunggah tidak akan muncul lagi.  
""")
