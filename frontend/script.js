/**
 * script.js — Vehicle Recognition AI Frontend Logic
 * 
 * Handles:
 * - Drag & drop image upload
 * - File validation
 * - API communication
 * - Results rendering with animations
 */

// ---- DOM Elements ----
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const btnRemove = document.getElementById('btnRemove');
const btnAnalyze = document.getElementById('btnAnalyze');
const uploadSection = document.getElementById('uploadSection');
const resultsSection = document.getElementById('resultsSection');
const statusText = document.getElementById('statusText');

// Result elements
const annotatedImage = document.getElementById('annotatedImage');
const vehicleModel = document.getElementById('vehicleModel');
const vehicleConfBar = document.getElementById('vehicleConfBar');
const vehicleConf = document.getElementById('vehicleConf');
const top5List = document.getElementById('top5List');
const plateText = document.getElementById('plateText');
const plateConf = document.getElementById('plateConf');
const plateBbox = document.getElementById('plateBbox');
const btnNew = document.getElementById('btnNew');

let selectedFile = null;

// ---- Upload Zone Events ----

// Click to browse
uploadZone.addEventListener('click', () => {
    fileInput.click();
});

// File selected via input
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Remove image
btnRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

// Analyze button
btnAnalyze.addEventListener('click', () => {
    if (selectedFile) {
        analyzeImage(selectedFile);
    }
});

// New analysis button
btnNew.addEventListener('click', () => {
    resetAll();
});

// ---- File Handling ----

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file (JPG, PNG, WEBP)');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showError('File too large. Maximum size is 10MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadZone.style.display = 'none';
        previewContainer.style.display = 'block';
        btnAnalyze.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadZone.style.display = 'block';
    previewContainer.style.display = 'none';
    btnAnalyze.style.display = 'none';
}

function resetAll() {
    resetUpload();
    resultsSection.style.display = 'none';
    uploadSection.style.display = 'block';
    setStatus('Ready', 'ready');
}

// ---- API Communication ----

async function analyzeImage(file) {
    const btnText = btnAnalyze.querySelector('.btn-text');
    const btnLoader = btnAnalyze.querySelector('.btn-loader');

    // Show loading state
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';
    btnAnalyze.disabled = true;
    setStatus('Analyzing...', 'processing');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }

        const result = await response.json();

        if (result.success) {
            displayResults(result);
            setStatus('Complete', 'ready');
        } else {
            throw new Error('Analysis returned no results');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showError(`Analysis failed: ${error.message}`);
        setStatus('Error', 'error');
    } finally {
        // Reset button
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        btnAnalyze.disabled = false;
    }
}

// ---- Results Display ----

function displayResults(result) {
    // Hide upload, show results
    uploadSection.style.display = 'none';
    resultsSection.style.display = 'grid';

    // Annotated image
    if (result.annotated_image) {
        annotatedImage.src = result.annotated_image;
    }

    // Vehicle model
    vehicleModel.textContent = result.vehicle_model || 'Unknown';
    const confPercent = Math.round((result.vehicle_confidence || 0) * 100);
    vehicleConf.textContent = `${confPercent}%`;

    // Animate confidence bar
    setTimeout(() => {
        vehicleConfBar.style.width = `${confPercent}%`;
    }, 100);

    // Top 5 predictions
    top5List.innerHTML = '';
    if (result.top5_predictions && result.top5_predictions.length > 0) {
        result.top5_predictions.forEach((pred, idx) => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="rank">${idx + 1}.</span>
                <span class="name">${pred.model}</span>
                <span class="conf">${Math.round(pred.confidence * 100)}%</span>
            `;
            top5List.appendChild(li);
        });
    }

    // License plate
    plateText.textContent = result.plate_text || 'Not detected';
    plateConf.textContent = result.plate_confidence
        ? `${Math.round(result.plate_confidence * 100)}%`
        : '—';
    plateBbox.textContent = result.plate_bbox
        ? `[${result.plate_bbox.join(', ')}]`
        : '—';
}

// ---- Utility ----

function setStatus(text, state) {
    statusText.textContent = text;
    const badge = statusText.parentElement;
    const dot = badge.querySelector('.badge-dot');

    badge.style.borderColor = state === 'error'
        ? 'rgba(239, 68, 68, 0.3)'
        : state === 'processing'
            ? 'rgba(245, 158, 11, 0.3)'
            : 'rgba(16, 185, 129, 0.2)';

    badge.style.background = state === 'error'
        ? 'rgba(239, 68, 68, 0.1)'
        : state === 'processing'
            ? 'rgba(245, 158, 11, 0.1)'
            : 'rgba(16, 185, 129, 0.1)';

    statusText.style.color = state === 'error'
        ? '#ef4444'
        : state === 'processing'
            ? '#f59e0b'
            : '#10b981';

    dot.style.background = state === 'error'
        ? '#ef4444'
        : state === 'processing'
            ? '#f59e0b'
            : '#10b981';
}

function showError(message) {
    alert(message); // Simple for now — could be replaced with a toast notification
}

// ---- Initialize ----
console.log('🚗 Vehicle Recognition AI — Frontend loaded');
