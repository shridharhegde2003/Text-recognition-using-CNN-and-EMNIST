document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const predictBtn = document.getElementById('predict-btn');
    const predictionOutput = document.getElementById('prediction-output');
    const predictionResult = document.getElementById('prediction-result');
    const confidenceScore = document.getElementById('confidence-score');
    const uploadTabBtn = document.getElementById('upload-tab');
    const drawTabBtn = document.getElementById('draw-tab');

    // --- Upload Elements ---
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');

    // --- Canvas Elements ---
    const canvas = document.getElementById('drawing-canvas');
    const clearCanvasBtn = document.getElementById('clear-canvas-btn');
    const ctx = canvas.getContext('2d');
    let drawing = false;
    let isCanvasDirty = false;

    // --- Canvas Initialization ---
    function clearCanvas() {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "black";
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        isCanvasDirty = false;
        predictBtn.disabled = true;
        predictionOutput.classList.add('d-none');
    }
    clearCanvas(); // Initial clear

    // --- Event Listeners ---
    const getPos = (evt) => {
        const rect = canvas.getBoundingClientRect();
        const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
        const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    };

    const startDrawing = (e) => {
        e.preventDefault();
        drawing = true;
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    };

    const draw = (e) => {
        if (!drawing) return;
        e.preventDefault();
        isCanvasDirty = true;
        predictBtn.disabled = false;
        const pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    };

    const stopDrawing = () => {
        drawing = false;
    };

    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);

    clearCanvasBtn.addEventListener('click', clearCanvas);

    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('d-none');
            };
            reader.readAsDataURL(file);
            predictBtn.disabled = false;
            predictionOutput.classList.add('d-none');
        }
    });

    // --- Prediction Logic ---
    predictBtn.addEventListener('click', () => {
        let activeTab = document.querySelector('.nav-link.active').id;
        const formData = new FormData();

        if (activeTab === 'upload-tab' && imageUpload.files[0]) {
            formData.append('file', imageUpload.files[0]);
        } else if (activeTab === 'draw-tab' && isCanvasDirty) {
            canvas.toBlob((blob) => {
                formData.append('file', blob, 'drawing.png');
                sendPrediction(formData);
            }, 'image/png');
            return; // sendPrediction is called in the callback
        } else {
            return; // No input to predict
        }
        sendPrediction(formData);
    });

    function sendPrediction(formData) {
        predictionResult.textContent = 'Predicting...';
        predictionOutput.classList.remove('d-none');
        confidenceScore.textContent = '';
        predictBtn.disabled = true;

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
            predictBtn.disabled = false;
        });
    }
});
