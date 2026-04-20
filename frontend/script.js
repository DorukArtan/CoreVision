/**
 * script.js — CoreVision Frontend Logic
 * 
 * Handles:
 * - Image & video upload with drag & drop
 * - File validation
 * - API communication (/predict and /predict-video)
 * - Dynamic results rendering for multi-vehicle pipeline responses
 */

// ---- DOM Elements ----
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const previewVideo = document.getElementById('previewVideo');
const previewFileInfo = document.getElementById('previewFileInfo');
const btnRemove = document.getElementById('btnRemove');
const btnAnalyze = document.getElementById('btnAnalyze');
const uploadSection = document.getElementById('uploadSection');
const resultsSection = document.getElementById('resultsSection');
const statusText = document.getElementById('statusText');
const tabImage = document.getElementById('tabImage');
const tabVideo = document.getElementById('tabVideo');
const uploadTitle = document.getElementById('uploadTitle');
const uploadDesc = document.getElementById('uploadDesc');
const uploadFormats = document.getElementById('uploadFormats');

// Result elements
const annotatedCard = document.getElementById('annotatedCard');
const annotatedImage = document.getElementById('annotatedImage');
const statsCard = document.getElementById('statsCard');
const statVehicles = document.getElementById('statVehicles');
const statPlates = document.getElementById('statPlates');
const statFrames = document.getElementById('statFrames');
const statFramesItem = document.getElementById('statFramesItem');
const vehiclesContainer = document.getElementById('vehiclesContainer');
const standalonePlatesContainer = document.getElementById('standalonePlatesContainer');
const btnNew = document.getElementById('btnNew');

let selectedFile = null;
let mediaType = 'image'; // 'image' or 'video'

// ---- Media Type Tabs ----

tabImage.addEventListener('click', () => switchMediaType('image'));
tabVideo.addEventListener('click', () => switchMediaType('video'));

function switchMediaType(type) {
    mediaType = type;
    
    // Update tab appearance
    tabImage.classList.toggle('active', type === 'image');
    tabVideo.classList.toggle('active', type === 'video');
    
    // Update upload zone text
    if (type === 'image') {
        uploadTitle.textContent = 'Upload Vehicle Image';
        uploadDesc.textContent = 'Drag & drop a car photo here, or click to browse';
        uploadFormats.textContent = 'Supports JPG, PNG, WEBP — Max 10MB';
        fileInput.accept = 'image/*';
    } else {
        uploadTitle.textContent = 'Upload Vehicle Video';
        uploadDesc.textContent = 'Drag & drop a dashcam or vehicle video here, or click to browse';
        uploadFormats.textContent = 'Supports MP4, AVI, MOV — Max 100MB';
        fileInput.accept = 'video/*';
    }
    
    // Reset if a different type file was loaded
    if (selectedFile) {
        resetUpload();
    }
}

// ---- Upload Zone Events ----

uploadZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

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

btnRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

btnAnalyze.addEventListener('click', () => {
    if (selectedFile) {
        if (mediaType === 'video') {
            analyzeVideo(selectedFile);
        } else {
            analyzeImage(selectedFile);
        }
    }
});

btnNew.addEventListener('click', () => {
    resetAll();
});

// ---- File Handling ----

function handleFile(file) {
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');
    
    // Auto-detect media type from file
    if (isImage && mediaType !== 'image') {
        switchMediaType('image');
    } else if (isVideo && mediaType !== 'video') {
        switchMediaType('video');
    }
    
    // Validate
    if (mediaType === 'image' && !isImage) {
        showToast('Please upload an image file (JPG, PNG, WEBP)', 'error');
        return;
    }
    if (mediaType === 'video' && !isVideo) {
        showToast('Please upload a video file (MP4, AVI, MOV)', 'error');
        return;
    }
    
    const maxSize = mediaType === 'image' ? 10 * 1024 * 1024 : 100 * 1024 * 1024;
    if (file.size > maxSize) {
        const maxMB = maxSize / (1024 * 1024);
        showToast(`File too large. Maximum size is ${maxMB}MB.`, 'error');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    if (isImage) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            previewVideo.style.display = 'none';
        };
        reader.readAsDataURL(file);
    } else {
        const url = URL.createObjectURL(file);
        previewVideo.src = url;
        previewVideo.style.display = 'block';
        previewImage.style.display = 'none';
    }
    
    // File info
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    previewFileInfo.textContent = `${file.name} — ${sizeMB} MB`;
    
    uploadZone.style.display = 'none';
    previewContainer.style.display = 'block';
    btnAnalyze.style.display = 'block';
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewImage.style.display = 'none';
    previewVideo.src = '';
    previewVideo.style.display = 'none';
    previewFileInfo.textContent = '';
    uploadZone.style.display = 'block';
    previewContainer.style.display = 'none';
    btnAnalyze.style.display = 'none';
}

function resetAll() {
    resetUpload();
    resultsSection.style.display = 'none';
    uploadSection.style.display = 'block';
    vehiclesContainer.innerHTML = '';
    standalonePlatesContainer.innerHTML = '';
    setStatus('Ready', 'ready');
}

// ---- API Communication ----

async function analyzeImage(file) {
    const btnText = btnAnalyze.querySelector('.btn-text');
    const btnLoader = btnAnalyze.querySelector('.btn-loader');
    
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
            const error = await response.json().catch(() => ({ detail: 'Server error' }));
            throw new Error(error.detail || 'Analysis failed');
        }
        
        const result = await response.json();
        
        if (result.success) {
            displayImageResults(result);
            setStatus('Complete', 'ready');
        } else {
            throw new Error('Analysis returned no results');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showToast(`Analysis failed: ${error.message}`, 'error');
        setStatus('Error', 'error');
    } finally {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        btnAnalyze.disabled = false;
    }
}

async function analyzeVideo(file) {
    const btnText = btnAnalyze.querySelector('.btn-text');
    const btnLoader = btnAnalyze.querySelector('.btn-loader');
    
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';
    btnAnalyze.disabled = true;
    setStatus('Processing video...', 'processing');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/predict-video', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Server error' }));
            throw new Error(error.detail || 'Video analysis failed');
        }
        
        const result = await response.json();
        
        if (result.success) {
            displayVideoResults(result);
            setStatus('Complete', 'ready');
        } else {
            throw new Error('Video analysis returned no results');
        }
    } catch (error) {
        console.error('Video analysis error:', error);
        showToast(`Video analysis failed: ${error.message}`, 'error');
        setStatus('Error', 'error');
    } finally {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        btnAnalyze.disabled = false;
    }
}

// ---- Image Results Display ----

function displayImageResults(result) {
    uploadSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Annotated image
    if (result.annotated_image) {
        annotatedImage.src = result.annotated_image;
        annotatedCard.style.display = 'block';
    } else {
        annotatedCard.style.display = 'none';
    }
    
    // Stats
    statVehicles.textContent = result.total_vehicles || 0;
    statPlates.textContent = result.total_plates || 0;
    statFramesItem.style.display = 'none';
    
    // Vehicles
    vehiclesContainer.innerHTML = '';
    const vehicles = result.vehicles || [];
    
    if (vehicles.length === 0) {
        vehiclesContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🚫</div>
                <p class="empty-state-text">No vehicles detected in this image</p>
            </div>
        `;
    } else {
        vehicles.forEach((vehicle, idx) => {
            vehiclesContainer.appendChild(createVehicleCard(vehicle, idx));
        });
    }
    
    // Standalone plates
    standalonePlatesContainer.innerHTML = '';
    const standalonePlates = result.standalone_plates || [];
    if (standalonePlates.length > 0) {
        const card = document.createElement('div');
        card.className = 'standalone-plates-card';
        card.innerHTML = `
            <h3 class="result-card-title">
                <span class="result-icon">🔤</span> Standalone Plates
            </h3>
            <div class="standalone-plates-grid" id="standalonePlatesGrid"></div>
        `;
        standalonePlatesContainer.appendChild(card);
        
        const grid = card.querySelector('#standalonePlatesGrid');
        standalonePlates.forEach(plate => {
            grid.appendChild(createPlateItem(plate));
        });
    }
    
    // Trigger confidence bar animations
    requestAnimationFrame(() => {
        document.querySelectorAll('.confidence-bar[data-width]').forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    });
}

// ---- Video Results Display ----

function displayVideoResults(result) {
    uploadSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // No annotated image for video
    annotatedCard.style.display = 'none';
    
    // Stats
    const summary = result.summary || {};
    statVehicles.textContent = summary.total_unique_vehicles || 0;
    statPlates.textContent = summary.total_unique_plates || 0;
    statFrames.textContent = result.frames_analyzed || 0;
    statFramesItem.style.display = 'block';
    
    // Video Info Card
    vehiclesContainer.innerHTML = '';
    const videoInfo = result.video_info || {};
    const summaryCard = document.createElement('div');
    summaryCard.className = 'video-summary-card';
    
    summaryCard.innerHTML = `
        <h3 class="result-card-title">
            <span class="result-icon">📹</span> Video Analysis Summary
        </h3>
        <div class="video-info-grid">
            <div class="video-info-item">
                <span class="video-info-value">${videoInfo.fps || '—'}</span>
                <span class="video-info-label">Video FPS</span>
            </div>
            <div class="video-info-item">
                <span class="video-info-value">${videoInfo.duration ? videoInfo.duration.toFixed(1) + 's' : '—'}</span>
                <span class="video-info-label">Duration</span>
            </div>
            <div class="video-info-item">
                <span class="video-info-value">${videoInfo.total_frames || '—'}</span>
                <span class="video-info-label">Total Frames</span>
            </div>
            <div class="video-info-item">
                <span class="video-info-value">${videoInfo.frames_analyzed || result.frames_analyzed || '—'}</span>
                <span class="video-info-label">Analyzed</span>
            </div>
        </div>
    `;
    
    // Unique vehicles
    const uniqueVehicles = summary.unique_vehicles || [];
    if (uniqueVehicles.length > 0) {
        const section = document.createElement('div');
        section.style.marginBottom = '16px';
        section.innerHTML = `<div class="block-title" style="margin-bottom: 8px;">🚗 Unique Vehicles Detected</div>`;
        const ul = document.createElement('ul');
        ul.className = 'video-unique-list';
        uniqueVehicles.forEach(v => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="unique-car-name">${escapeHtml(v.make_model || 'Unknown')}</span>
                <span class="unique-car-conf">${v.confidence ? Math.round(v.confidence * 100) + '%' : '—'}</span>
            `;
            ul.appendChild(li);
        });
        section.appendChild(ul);
        summaryCard.appendChild(section);
    }
    
    // Unique plates
    const uniquePlates = summary.unique_plates || [];
    if (uniquePlates.length > 0) {
        const section = document.createElement('div');
        section.innerHTML = `<div class="block-title" style="margin-bottom: 8px;">🔤 Unique Plates Detected</div>`;
        const ul = document.createElement('ul');
        ul.className = 'video-unique-list';
        uniquePlates.forEach(p => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="unique-plate-text">${escapeHtml(p.text || '—')}</span>
                <span class="unique-plate-country">${escapeHtml(p.country || '')} ${p.associated_car ? '(' + escapeHtml(p.associated_car) + ')' : ''}</span>
            `;
            ul.appendChild(li);
        });
        section.appendChild(ul);
        summaryCard.appendChild(section);
    }
    
    if (uniqueVehicles.length === 0 && uniquePlates.length === 0) {
        summaryCard.innerHTML += `
            <div class="empty-state">
                <div class="empty-state-icon">🚫</div>
                <p class="empty-state-text">No vehicles or plates detected in this video</p>
            </div>
        `;
    }
    
    vehiclesContainer.appendChild(summaryCard);
}

// ---- Vehicle Card Builder ----

function createVehicleCard(vehicle, idx) {
    const card = document.createElement('div');
    card.className = 'vehicle-card';
    card.style.animationDelay = `${idx * 0.1}s`;
    
    // Header
    const vType = vehicle.vehicle_type || 'vehicle';
    const detConf = vehicle.vehicle_det_confidence
        ? `${Math.round(vehicle.vehicle_det_confidence * 100)}% detection`
        : '';
    
    let html = `
        <div class="vehicle-card-header">
            <div class="vehicle-number">
                <span class="vehicle-index">${idx + 1}</span>
                <span class="vehicle-type-badge">${escapeHtml(vType)}</span>
            </div>
            <span class="vehicle-det-conf">${detConf}</span>
        </div>
        <div class="vehicle-body">
    `;
    
    // Left column — Classification
    html += '<div class="classification-block">';
    html += '<div class="block-title">Car Classification</div>';
    
    // Brand badges
    const brands = [];
    if (vehicle.brand) {
        brands.push({ label: 'Brand', name: vehicle.brand, conf: vehicle.brand_confidence, cls: 'primary' });
    }
    if (vehicle.clip_brand) {
        brands.push({ label: 'CLIP', name: vehicle.clip_brand, conf: vehicle.clip_brand_confidence, cls: 'clip' });
    }
    
    if (brands.length > 0) {
        // Show the highest confidence brand as the main heading
        const best = brands.reduce((a, b) => (b.conf || 0) > (a.conf || 0) ? b : a);
        const bestConfPct = best.conf ? Math.round(best.conf * 100) : 0;
        html += `<div class="model-name">${escapeHtml(formatModelName(best.name))}</div>`;
        html += `
            <div class="confidence-row">
                <div class="confidence-bar-container">
                    <div class="confidence-bar" data-width="${bestConfPct}%"></div>
                </div>
                <span class="confidence-text">${bestConfPct}%</span>
            </div>
        `;
        html += '<div class="brand-badges">';
        brands.forEach(b => {
            const bConf = b.conf ? ` ${Math.round(b.conf * 100)}%` : '';
            html += `
                <span class="brand-badge ${b.cls}">
                    <span class="brand-badge-label">${b.label}:</span>
                    ${escapeHtml(b.name)}${bConf}
                </span>
            `;
        });
        html += '</div>';
    } else {
        html += '<div class="model-name">Unknown</div>';
    }
    
    html += '</div>'; // end classification-block
    
    // Right column — Plates
    html += '<div class="plates-block">';
    html += '<div class="block-title">License Plates</div>';
    
    const plates = vehicle.plates || [];
    if (plates.length === 0) {
        html += '<div class="no-plates">No plates detected for this vehicle</div>';
    } else {
        plates.forEach(plate => {
            html += createPlateItemHTML(plate);
        });
    }
    
    html += '</div>'; // end plates-block
    html += '</div>'; // end vehicle-body
    
    card.innerHTML = html;
    return card;
}

// ---- Plate Item Builder ----

function createPlateItemHTML(plate) {
    const text = plate.text || '';
    const ocrConf = plate.ocr_confidence;
    const bbox = plate.plate_bbox;
    const country = plate.country;
    const countryCode = plate.country_code;
    
    let html = '<div class="plate-item">';
    
    // Plate text display
    html += '<div class="plate-display">';
    if (text) {
        html += `<span class="plate-text">${escapeHtml(text)}</span>`;
    } else {
        html += '<span class="plate-text not-detected">No text detected</span>';
    }
    html += '</div>';
    
    // Meta info
    html += '<div class="plate-meta">';
    
    html += `
        <div class="plate-meta-item">
            <span class="plate-meta-label">OCR Confidence</span>
            <span class="plate-meta-value">${ocrConf != null ? Math.round(ocrConf * 100) + '%' : '—'}</span>
        </div>
    `;
    
    html += `
        <div class="plate-meta-item">
            <span class="plate-meta-label">Detection</span>
            <span class="plate-meta-value">${plate.plate_det_confidence != null ? Math.round(plate.plate_det_confidence * 100) + '%' : '—'}</span>
        </div>
    `;
    
    html += '</div>'; // end plate-meta
    
    // Country badge
    if (country && country !== 'Unknown') {
        const flag = countryCodeToFlag(countryCode);
        html += `
            <div style="text-align: center;">
                <span class="country-badge">
                    <span class="country-flag">${flag}</span>
                    ${escapeHtml(country)}
                </span>
            </div>
        `;
    }
    
    html += '</div>'; // end plate-item
    return html;
}

function createPlateItem(plate) {
    const div = document.createElement('div');
    div.innerHTML = createPlateItemHTML(plate);
    return div.firstElementChild;
}

// ---- Utility Functions ----

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

/**
 * Format a model name string for display.
 * Converts underscores to spaces and capitalizes words.
 */
function formatModelName(name) {
    if (!name || name === 'Unknown') return 'Unknown';
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * Convert a 2-letter country code to flag emoji.
 */
function countryCodeToFlag(code) {
    // Map vehicle registration codes to ISO 3166-1 alpha-2 for flag emoji
    const regToIso = {
        'D': 'DE', 'F': 'FR', 'I': 'IT', 'E': 'ES', 'A': 'AT', 'B': 'BE',
        'H': 'HU', 'L': 'LU', 'M': 'MT', 'N': 'NO', 'P': 'PT', 'S': 'SE',
        'V': 'VA', 'J': 'JP', 'T': 'TH', 'Q': 'QA',
        'RUS': 'RU', 'USA': 'US', 'CDN': 'CA', 'MEX': 'MX', 'AUS': 'AU',
        'IND': 'IN', 'ROK': 'KR', 'IRL': 'IE', 'EST': 'EE', 'FIN': 'FI',
        'SLO': 'SI', 'SRB': 'RS', 'BIH': 'BA', 'MNE': 'ME', 'MK': 'MK',
        'FL': 'LI', 'AND': 'AD', 'RSM': 'SM', 'RCH': 'CL', 'RA': 'AR',
        'RI': 'ID', 'RP': 'PH', 'VN': 'VN', 'EAK': 'KE', 'EAU': 'UG',
        'IRQ': 'IQ', 'SYR': 'SY', 'KWT': 'KW', 'BRN': 'BH', 'OM': 'OM',
        'UAE': 'AE', 'EU': 'EU',
    };
    if (!code) return '🏳️';
    const iso = regToIso[code.toUpperCase()] || code.toUpperCase();
    if (iso.length !== 2) return '🏳️';
    const codePoints = [...iso].map(
        c => 0x1F1E6 + c.charCodeAt(0) - 65
    );
    return String.fromCodePoint(...codePoints);
}

/**
 * Escape HTML special characters to prevent XSS.
 */
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Show a toast notification.
 */
function showToast(message, type = 'error') {
    // Remove existing toast
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    // Trigger animation
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });
    
    // Auto-dismiss
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ---- Initialize ----
console.log('🔍 CoreVision — Frontend loaded');
