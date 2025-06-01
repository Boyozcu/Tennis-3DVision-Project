# Tenis Maçı Analizi Projesi: Temel Bilgisayarlı Görü ile Otomatik Takip ve İstatistik Çıkarımı (BLM5103)

## Proje Amacı

Bu proje, data/tennis.mp4 video dosyasını kullanarak tenis maçı analizi yapmayı hedefler. Sadece klasik Bilgisayar Görüşü algoritmaları ile oyuncu ve top tespiti, takibi, kort geometrisi analizi ve kuşbakışı haritalaması gerçekleştirilecek; elde edilen verilerden oyun istatistikleri çıkarılacak ve sonuçlar görselleştirilecektir.

## Ana Kurallar ve Kısıtlamalar

- **Tek Dosya Kuralı**: Tüm kod, tek bir Python (tennis_analysis.py) dosyası içinde olmalıdır
- **Algoritma Kısıtlaması**: Yalnızca klasik Bilgisayar Görüşü algoritmaları kullanılacaktır. Makine Öğrenimi veya Derin Öğrenme yöntemleri KESİNLİKLE KULLANILMAYACAKTIR
- **Girdi Videosu**: data/tennis.mp4 (1280x720 çözünürlük, 60 FPS)
- **Tespit Hedefi**: Geçerli her karede tam olarak 2 oyuncu ve 1 top tespit edilmelidir
- **Kullanıcı Onayı**: Tam video işlemeye başlamadan önce, ilk 2 başarılı tespit karesi kullanıcıya gösterilmeli ve onay istenmelidir

## Kullanılan Teknikler

### Kort Tespiti ve Homografi
- Canny kenar tespiti
- Hough Transform ile çizgi tespiti
- Homografi hesaplaması

### Oyuncu Tespiti ve Takibi
- MOG2 arka plan çıkarımı
- Morfolojik operasyonlar
- Kontur analizi ve filtreleme
- Alan, en-boy oranı ve sağlamlık kriterleri

### Top Tespiti ve Takibi
- HSV renk uzayında filtreleme
- Dairesellik analizi
- Kalman filtresi ile tahmin
- Kademeli arama ve yeniden kazanım

### Görselleştirme
- Ana video üzerine tespit katmanları
- Kuşbakışı kort görünümü
- Oyun istatistikleri analizi

## Çalıştırma

```bash
python tennis_analysis.py
```

## Gereksinimler

- Python 3.x
- OpenCV
- NumPy

## Çıktılar

- `output.mp4`: Tespit katmanları ile ana video
- `sketch_output.mp4`: Kuşbakışı kort görünümü
- Konsol çıktısı: Oyun istatistikleri

## Özellikler

- **Gerçek zamanlı tespit**: Video karelerde oyuncu ve top tespiti
- **Homografi mapping**: 2D görüntüden kuşbakışı kort görünümü
- **Takip algoritmaları**: Kalman filtresi ile düzgün top takibi
- **İstatistik analizi**: Ralli sayısı, vuruş yönü değişimleri, kort bölgesi analizi
- **Görsel çıktı**: İki farklı video çıktısı ile detaylı analiz

## Proje Yapısı

```
Tennis-3DVision-Project/
├── tennis_analysis.py          # Ana analiz dosyası
├── data/
│   └── tennis.mp4             # Girdi video dosyası
├── output.mp4                 # Ana çıktı videosu
├── sketch_output.mp4          # Kuşbakışı çıktı videosu
└── README.md                  # Bu dosya
```
