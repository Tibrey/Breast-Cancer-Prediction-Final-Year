

// // document.addEventListener("DOMContentLoaded", function () {

// //     const form = document.getElementById('image-form');
// //     const processButton = document.getElementById('image-process');

// //     const dropArea = document.getElementById('drop-area');
// //     const inputFile = document.getElementById('input-file');
// //     const imgView = document.getElementById('img-view');

// //     inputFile.addEventListener('change', uploadImage);

// //     function uploadImage() {

// //         let imgLink = URL.createObjectURL(inputFile.files[0]);
// //         imgView.style.backgroundImage = `url(${imgLink})`;

// //         imgView.textContent = '';
// //     }

// //     dropArea.addEventListener('dragover', function (e) {
// //         e.preventDefault();
// //     })

// //     dropArea.addEventListener('drop', function (e) {
// //         e.preventDefault();
// //         inputFile.files = e.dataTransfer.files;
// //         uploadImage();
// //     })


// // });

// document.addEventListener("DOMContentLoaded", function () {
//     const form = document.getElementById('image-form');
//     const processButton = document.getElementById('image-process');
//     const dropArea = document.getElementById('drop-area');
//     const inputFile = document.getElementById('input-file');
//     const imgView = document.getElementById('img-view');
//     const probabilityGraph = document.getElementById('probability-graph'); // Add this line

//     inputFile.addEventListener('change', uploadImage);

//     function uploadImage() {
//         let imgLink = URL.createObjectURL(inputFile.files[0]);
//         imgView.style.backgroundImage = `url(${imgLink})`;
//         imgView.textContent = '';

//         // Reset the probability graph when a new image is uploaded
//         probabilityGraph.src = '';
//     }

//     dropArea.addEventListener('dragover', function (e) {
//         e.preventDefault();
//     })

//     dropArea.addEventListener('drop', function (e) {
//         e.preventDefault();
//         inputFile.files = e.dataTransfer.files;
//         uploadImage();
//     })
//         processButton.addEventListener('click', function (e) {
//         // Manually trigger the form submission when the processButton is clicked
//         form.submit();
//     });

//     form.addEventListener('submit', function (e) {
//         // Prevent the form from submitting to see the dynamic update
//         e.preventDefault();

//         // Simulate receiving the base64-encoded image data from the server
//         const base64Data = 'your_base64_image_data_here'; // Replace with actual data

//         // Update the probability graph
//         probabilityGraph.src = 'data:image/png;base64,' + base64Data;
//     });
// });


// document.addEventListener("DOMContentLoaded", function () {
//     const form = document.getElementById('image-form');
//     const processButton = document.getElementById('image-process');
//     const dropArea = document.getElementById('drop-area');
//     const inputFile = document.getElementById('input-file');
//     const imgView = document.getElementById('img-view');
//     const probabilityGraph = document.getElementById('probability-graph');
//     const predictionResult = document.getElementById('prediction-result'); // Add this line

//     inputFile.addEventListener('change', uploadImage);

//     function uploadImage() {
//         let imgLink = URL.createObjectURL(inputFile.files[0]);
//         imgView.style.backgroundImage = `url(${imgLink})`;
//         imgView.textContent = '';

//         // Reset the probability graph and prediction result when a new image is uploaded
//         probabilityGraph.src = '';
//         predictionResult.textContent = '';
//     }

//     dropArea.addEventListener('dragover', function (e) {
//         e.preventDefault();
//     });

//     dropArea.addEventListener('drop', function (e) {
//         e.preventDefault();
//         inputFile.files = e.dataTransfer.files;
//         uploadImage();
//     });

//     processButton.addEventListener('click', function (e) {
//         // Manually trigger the form submission when the processButton is clicked
//         form.submit();

//         form.addEventListener('submit', function (e) {
//             // Prevent the form from submitting to see the dynamic update
//             e.preventDefault();

//             // Simulate receiving the base64-encoded image data and prediction data from the server
//             const base64Data = 'your_base64_image_data_here'; // Replace with actual data
//             const predictionData = 'Your prediction result here'; // Replace with actual data

//             // Update the probability graph and prediction result
//             probabilityGraph.src = 'data:image/png;base64,' + base64Data;
//             predictionResult.textContent = 'Prediction Result: ' + predictionData;

//             console.log('Form submitted');
//         });
//     });


document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById('image-form');
    const processButton = document.getElementById('image-process');
    const dropArea = document.getElementById('drop-area');
    const inputFile = document.getElementById('input-file');
    const imgView = document.getElementById('img-view');
    const probabilityGraph = document.getElementById('probability-graph'); // Add this line
    const probabilityList = document.getElementById('probability-list'); // Add this line

    inputFile.addEventListener('change', uploadImage);

    function uploadImage() {
        let imgLink = URL.createObjectURL(inputFile.files[0]);
        imgView.style.backgroundImage = `url(${imgLink})`;
        imgView.textContent = '';

        // Reset the probability graph and list when a new image is uploaded
        probabilityGraph.src = '';
        probabilityList.innerHTML = '';
    }

    dropArea.addEventListener('dragover', function (e) {
        e.preventDefault();
    })

    dropArea.addEventListener('drop', function (e) {
        e.preventDefault();
        inputFile.files = e.dataTransfer.files;
        uploadImage();
    })

    processButton.addEventListener('click', function (e) {
        // Manually trigger the form submission when the processButton is clicked
        form.submit();
    });

    form.addEventListener('submit', function (e) {
        // Prevent the form from submitting to see the dynamic update
        e.preventDefault();

        // Simulate receiving the base64-encoded image data and probability values from the server
        const base64Data = 'your_base64_image_data_here'; // Replace with actual data
        const probValues = { label1: 0.8, label2: 0.2 }; // Replace with actual probability values

        // Update the probability graph
        probabilityGraph.src = 'data:image/png;base64,' + base64Data;

        // Update the probability list
        probabilityList.innerHTML = Object.entries(probValues).map(([label, probability]) =>
            `<li>${label}: ${(probability * 100).toFixed(2)}%</li>`
        ).join('');
    });
});
