
const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const resultDiv = document.getElementById("result");


form.addEventListener("submit", async (e) => {
  e.preventDefault();


  if (!fileInput.files[0]) {
    resultDiv.textContent = "Lütfen önce bir dosya seçin.";
    return;
  }

  resultDiv.textContent = "Analiz ediliyor…";

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (data.error) {
      resultDiv.textContent = "Hata: " + data.error;
    } else {
      resultDiv.innerHTML = `
        <p><strong>Tahmin:</strong> ${data.prediction}</p>
        <p><strong>Güven:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
      `;
    }
  } catch (err) {
    console.error(err);
    resultDiv.textContent = "Sunucuya bağlanırken bir hata oluştu.";
  }
});
