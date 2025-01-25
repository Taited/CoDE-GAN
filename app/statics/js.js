const imageCanvas = document.getElementById('image-layer');
const maskCanvas = document.getElementById('mask-layer');
const sketchCanvas = document.getElementById('sketch-layer');
const drawingCanvas = document.getElementById('drawing-layer');

const imageCtx = imageCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');
const sketchCtx = sketchCanvas.getContext('2d');
const drawingCtx = drawingCanvas.getContext('2d');

let currentCtx = maskCtx;
let drawing = false;
let currentLayer = 'mask';
let maskCurrentThickness = 20;
let sketchCurrentThickness = 3;

const alpha = currentLayer === 'mask' ? 0.5 : 1;
drawingCtx.globalAlpha = alpha;


document.getElementById('mask-thickness').addEventListener('input', function (event) {
    maskCurrentThickness = event.target.value;
});


// mouse drawing
drawingCanvas.addEventListener('mousedown', startDrawing);
drawingCanvas.addEventListener('mousemove', draw);
drawingCanvas.addEventListener('mouseup', stopDrawing);

// touch drawing
drawingCanvas.addEventListener('touchstart', startDrawingTouch);
drawingCanvas.addEventListener('touchmove', drawTouch);
drawingCanvas.addEventListener('touchend', stopDrawing);

function startDrawing(event) {
    drawing = true;
    const rect = drawingCanvas.getBoundingClientRect();
    const scaleX = drawingCanvas.width / rect.width;
    const scaleY = drawingCanvas.height / rect.height;
    drawingCtx.beginPath();
    drawingCtx.moveTo((event.clientX - rect.left) * scaleX, (event.clientY - rect.top) * scaleY);
}

function startDrawingTouch(event) {
    drawing = true;
    const rect = drawingCanvas.getBoundingClientRect();
    const scaleX = drawingCanvas.width / rect.width;
    const scaleY = drawingCanvas.height / rect.height;
    drawingCtx.beginPath();
    // Use the first touch point
    const touch = event.touches[0];
    drawingCtx.moveTo((touch.clientX - rect.left) * scaleX, (touch.clientY - rect.top) * scaleY);
    event.preventDefault();
}

function draw(event) {
    if (!drawing) return;
    const rect = drawingCanvas.getBoundingClientRect();
    const scaleX = drawingCanvas.width / rect.width;
    const scaleY = drawingCanvas.height / rect.height;
    drawingCtx.globalAlpha = 1.0;
    drawingCtx.globalCompositeOperation = 'destination-atop';
    drawingCtx.lineTo((event.clientX - rect.left) * scaleX, (event.clientY - rect.top) * scaleY);
    drawingCtx.lineWidth = currentLayer === 'mask' ? maskCurrentThickness : sketchCurrentThickness;
    drawingCtx.strokeStyle = currentLayer === 'mask' ? 'rgb(10,10,10)' : 'rgb(255,255,255)';
    drawingCtx.lineCap = 'round';
    drawingCtx.lineJoin = 'round';
    drawingCtx.stroke();
}

function drawTouch(event) {
    if (!drawing) return;
    const rect = drawingCanvas.getBoundingClientRect();
    const scaleX = drawingCanvas.width / rect.width;
    const scaleY = drawingCanvas.height / rect.height;
    drawingCtx.globalAlpha = 1.0;
    drawingCtx.globalCompositeOperation = 'destination-atop';
    // Use the first touch point
    const touch = event.touches[0];
    drawingCtx.lineTo((touch.clientX - rect.left) * scaleX, (touch.clientY - rect.top) * scaleY);
    drawingCtx.lineWidth = currentLayer === 'mask' ? maskCurrentThickness : sketchCurrentThickness;
    drawingCtx.strokeStyle = currentLayer === 'mask' ? 'rgb(10,10,10)' : 'rgb(255,255,255)';
    drawingCtx.lineCap = 'round';
    drawingCtx.lineJoin = 'round';
    drawingCtx.stroke();
    event.preventDefault();
}

// begin
let tempCanvas = document.createElement('canvas');
let tempCtx = tempCanvas.getContext('2d');
// Set the size of the temporary canvas to match the mask canvas
tempCanvas.width = maskCanvas.width;
tempCanvas.height = maskCanvas.height;

// Set the position of the temporary canvas to match the mask canvas
tempCanvas.style.position = 'absolute';
tempCanvas.style.left = maskCanvas.offsetLeft + 'px';
tempCanvas.style.top = maskCanvas.offsetTop + 'px';

function stopDrawing(event) {
    event.preventDefault();
    drawing = false;
    if (currentLayer === 'mask') {
        tempCtx.drawImage(drawingCanvas, 0, 0);
        currentCtx.globalAlpha = 0.7;
        currentCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        currentCtx.drawImage(tempCanvas, 0, 0);
    }
    else {
        currentCtx.drawImage(drawingCanvas, 0, 0);
    }
    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    imageCtx.drawImage(imageCanvas, 0, 0);
    currentCtx.globalAlpha = 1.0;
}

document.getElementById('layer').addEventListener('change', function (event) {
    currentLayer = event.target.value;
    currentCtx = currentLayer === 'mask' ? maskCtx : sketchCtx;
});


// upload image
document.getElementById('image-input').addEventListener('input', function (event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
            imageCtx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
});

// get image from backend
async function random(){
    showLoader('drawing-layer')
    try {
        const randomUrl = `${window.location.origin}/code-gan/random-image/?random=` + Math.random();

        const canvas = document.getElementById('image-layer');
        const context = canvas.getContext('2d');
        const response = await fetch(randomUrl);
        if (response.ok)
        {
            blob = await response.blob();
            const imageURL = URL.createObjectURL(blob);
            // load image to canvas
            const image = new Image();
            image.onload = function () {
                context.drawImage(image, 0, 0, canvas.width, canvas.height);
                URL.revokeObjectURL(imageURL); // release url object
            };
            image.src = imageURL;
        }
        else {
            console.error('Error uploading images:', response.statusText);
        }
    } catch (error) {
        console.error('Error:', error);
    } finally {
        hideLoader();
    }
}


function showLoader(elementID) {
    const drawingLayer1 = document.getElementById(elementID);
    const rect = drawingLayer1.getBoundingClientRect();

    const loaderBackground = document.createElement('div');
    loaderBackground.id = 'loader-background';
    loaderBackground.style.position = 'absolute';
    loaderBackground.style.top = rect.top + 'px';
    loaderBackground.style.left = rect.left + 'px';
    loaderBackground.style.width = rect.width + 'px';
    loaderBackground.style.height = rect.height + 'px';
    loaderBackground.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    loaderBackground.style.display = 'flex';
    loaderBackground.style.justifyContent = 'center';
    loaderBackground.style.alignItems = 'center';
    document.body.appendChild(loaderBackground);

    const loader = document.createElement('div');
    loader.id = 'loader';
    loaderBackground.appendChild(loader);
}


function hideLoader() {
    const loaderBackground = document.getElementById('loader-background');
    if (loaderBackground) {
        loaderBackground.parentNode.removeChild(loaderBackground);
    }
}

async function inference(){
    showLoader('drawing-layer1');
    const formData = new FormData();
    // Convert canvas content to Blob objects
    const imgBlob = await canvasToBlob(imageCanvas, 'image/png');
    const sketchBlob = await canvasToBlob(sketchCanvas, 'image/png');
    const maskBlob = await canvasToBlob(maskCanvas, 'image/png');

    formData.append('img', imgBlob, 'img.png');
    formData.append('sketch', sketchBlob, 'sketch.png');
    formData.append('mask', maskBlob, 'mask.png');

    try {
        const apiUrl = `${window.location.origin}/code-gan/process-images/`;

        const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData,
        });

        // Handle the response
        if (response.ok) {
            const blob = await response.blob();
            const imageURL = URL.createObjectURL(blob);
            const outputLayer = document.getElementById('drawing-layer1');
            const ctx = outputLayer.getContext('2d');
            const img = new Image();
            img.src = imageURL;
            img.onload = () => {
                ctx.drawImage(img, 0, 0, outputLayer.width, outputLayer.height);
            };
        } else {
            console.error('Error uploading images:', response.statusText);
        }
    } catch (error) {
        console.error('Error:', error);
    } finally {
        hideLoader();
    }
}

function canvasToBlob(canvas, mimeType) {
    return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) {
                resolve(blob);
            } else {
                reject(new Error('Canvas to Blob conversion failed.'));
            }
        }, mimeType);
    });
}

function clearCanvas() {
    // get all canvas context
    const imageCtx = imageCanvas.getContext('2d');
    const maskCtx = maskCanvas.getContext('2d');
    const sketchCtx = sketchCanvas.getContext('2d');
    const drawingCtx = drawingCanvas.getContext('2d');
    const drawing1Canvas = document.getElementById('drawing-layer1')
    const outputCtx = drawing1Canvas.getContext('2d');

    // clear all context
    imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    sketchCtx.clearRect(0, 0, sketchCanvas.width, sketchCanvas.height);
    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    outputCtx.clearRect(0, 0, drawing1Canvas.width, drawing1Canvas.height);
    tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height)

    // clear the file input value
    document.getElementById('image-input').value = '';
}
