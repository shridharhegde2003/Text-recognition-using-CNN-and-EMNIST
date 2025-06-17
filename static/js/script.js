document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const predictBtn = document.getElementById('predict-btn');
    const predictionOutput = document.getElementById('prediction-output');
    const predictionResult = document.getElementById('prediction-result');
    const confidenceScore = document.getElementById('confidence-score');

    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // Display image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('d-none');
            };
            reader.readAsDataURL(file);

            // Enable the predict button and hide old results
            predictBtn.disabled = false;
            predictionOutput.classList.add('d-none');
            predictionResult.textContent = '';
            confidenceScore.textContent = '';
        }
    });

    predictBtn.addEventListener('click', () => {
        const file = imageUpload.files[0];
        if (!file) {
            return;
        }

        // Show loading state
        predictionResult.textContent = 'Predicting...';
        predictionOutput.classList.remove('d-none');
        predictBtn.disabled = true; // Disable button during request

        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                predictionResult.textContent = `Error: ${data.error}`;
                predictionResult.classList.remove('text-success');
                predictionResult.classList.add('text-danger');
                confidenceScore.textContent = '';
            } else {
                predictionResult.textContent = `Predicted: ${data.prediction}`;
                confidenceScore.textContent = `Confidence: ${data.confidence}%`;
                predictionResult.classList.remove('text-danger');
                predictionResult.classList.add('text-success');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            predictionResult.textContent = 'An error occurred.';
            predictionResult.classList.remove('text-success');
            predictionResult.classList.add('text-danger');
        })
        .finally(() => {
            // Re-enable button if user wants to try another file
            predictBtn.disabled = false;
        });
    });
});
