// Selecting elements
const cameraButton = document.getElementById('cameraButton');
const uploadButton = document.getElementById('uploadButton');
const video = document.getElementById('video');
const imageUpload = document.getElementById('imageUpload');
const resultCanvas = document.getElementById('resultCanvas');
const ctx = resultCanvas.getContext('2d');

// Event listener for Camera button
cameraButton.addEventListener('click', async () => {
    // Show the video element
    video.style.display = 'block';
    resultCanvas.style.display = 'none';

    // Ask for permission to access the camera
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        console.error('Error accessing camera:', err);
    }
});

// Event listener for Upload button
uploadButton.addEventListener('click', () => {
    // Trigger the hidden file input for image upload
    imageUpload.click();
});

// Event listener for Image Upload
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Draw the uploaded image on the canvas
                resultCanvas.style.display = 'block';
                video.style.display = 'none';
                ctx.drawImage(img, 0, 0, resultCanvas.width, resultCanvas.height);
                // Send the image to the server for prediction
                sendImageToServer(resultCanvas.toDataURL());
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// Capture a photo from the video stream when the user clicks on the canvas
video.addEventListener('click', () => {
    // Draw the current frame from the video onto the canvas
    ctx.drawImage(video, 0, 0, resultCanvas.width, resultCanvas.height);
    resultCanvas.style.display = 'block';
    video.style.display = 'none';
    // Send the captured image to the server for prediction
    sendImageToServer(resultCanvas.toDataURL());
});

// Function to send the image data to the server for prediction
function sendImageToServer(imageData) {
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
    })
    .then(response => response.json())
    .then(data => {
        alert(`Emotion: ${data.emotion}, Age: ${data.age}, Gender: ${data.gender}, Ethnicity: ${data.ethnicity}`);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while predicting.');
    });
}
