document.addEventListener("DOMContentLoaded", function () {
  const labels = document.querySelectorAll(".form_container > label");

  labels.forEach((label) => {
    const input = label.querySelector("input");
    const span = label.querySelector("span");

    input.addEventListener("focus", () => {
      label.classList.add("focus");
      span.style.top = "0";
    });

    input.addEventListener("blur", () => {
      if (!input.value.trim()) {
        label.classList.remove("focus");
        span.style.top = "50%";
      }
    });
  });
});

document.addEventListener("DOMContentLoaded", function () {
  function validateInputs() {
    var radiusMean = parseFloat(document.getElementById("feature0").value);
    var perimeterMean = parseFloat(document.getElementById("feature2").value);
    var areaMean = parseFloat(document.getElementById("feature3").value);
    var concavePointsMean = parseFloat(
      document.getElementById("feature7").value
    );
    var radiusWorst = parseFloat(document.getElementById("feature15").value);
    var perimeterWorst = parseFloat(document.getElementById("feature17").value);
    var areaWorst = parseFloat(document.getElementById("feature18").value);
    var concavePointsWorst = parseFloat(
      document.getElementById("feature22").value
    );

    var valid = true;

    // Validate Radius Mean
    if (isNaN(radiusMean) || radiusMean < 6.981 || radiusMean > 28.11) {
      document.getElementById("feature0").style.borderColor = "red";
      valid = false;
    } else {
      document.getElementById("feature0").style.borderColor = "";
    }

    // Validate Perimeter Mean
    if (
      isNaN(perimeterMean) ||
      perimeterMean < 43.79 ||
      perimeterMean > 188.5
    ) {
      document.getElementById("feature2").style.borderColor = "red";
      valid = false;
    } else {
      document.getElementById("feature2").style.borderColor = "";
    }

    // Validate Area Mean
    if (isNaN(areaMean) || areaMean < 143.5 || areaMean > 2501) {
      document.getElementById("feature3").style.borderColor = "red";
      valid = false;
    } else {
      document.getElementById("feature3").style.borderColor = "";
    }

    // Validate Concave Points Mean
    if (
      isNaN(concavePointsMean) ||
      concavePointsMean < 0 ||
      concavePointsMean > 0.2012
    ) {
      document.getElementById("feature7").style.borderColor = "red";
      valid = false;
    } else {
      document.getElementById("feature7").style.borderColor = "";
    }

    // Validate Radius Worst
    if (isNaN(radiusWorst) || radiusWorst < 7.93 || radiusWorst > 36.04) {
      document.getElementById("feature15").style.borderColor = "red";
      valid = false;
    } else {
      document.getElementById("feature15").style.borderColor = "";
    }

    // Validate Perimeter Worst
    if (
      isNaN(perimeterWorst) ||
      perimeterWorst < 50.41 ||
      perimeterWorst > 251.2
    ) {
      document.getElementById("feature17").style.borderColor = "red";
      valid = false;
    } else {
      document.getElementById("feature17").style.borderColor = "";
    }

    // Validate Area Worst
    if (isNaN(areaWorst) || areaWorst < 185.2 || areaWorst > 4254) {
      document.getElementById("feature18").style.borderColor = "red";
      valid = false;
    } else {
      document.getElementById("feature18").style.borderColor = "";
    }

    // Validate Concave Points Worst
    if (
      isNaN(concavePointsWorst) ||
      concavePointsWorst < 0 ||
      concavePointsWorst > 0.291
    ) {
      document.getElementById("feature22").style.borderColor = "red";
      valid = false;
    } else {
      document.getElementById("feature22").style.borderColor = "";
    }

    if (!valid) {
      alert("Value must be within the range");
    }

    return valid;
  }

  var form = document.querySelector(".form");
  form.onsubmit = function () {
    return validateInputs();
  };

  function clearPrediction() {
    var modelSelect = document.getElementById("model");
    var predictionResult = document.getElementById("predictionResult");
    if (modelSelect.value === "please_select_model") {
      predictionResult.innerHTML = ""; // Clear prediction result
    }
  }
});

document.addEventListener("DOMContentLoaded", function () {
  document.getElementById("model").addEventListener("change", function () {
    var selectedModel = this.value;
    var featureForm = document.getElementById("featureForm");
    if (selectedModel === "Logistic regression" || selectedModel === "SVM") {
      featureForm.style.display = "block";
    } else {
      featureForm.style.display = "none";
    }
  });
});
$(document).ready(function () {
  $("#predictForm").submit(function (event) {
    event.preventDefault();
    var formData = $(this).serialize();

    $.ajax({
      type: "POST",
      url: "/predict",
      data: formData,
      success: function (response) {
        $("#prediction").text(response.prediction);
      },
    });
  });
});

document.addEventListener("DOMContentLoaded", function () {
  var modelSelect = document.getElementById("model");
  var predictionDiv = document.getElementById("prediction");

  modelSelect.addEventListener("change", function () {
    var selectedModel = modelSelect.value;

    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: "model=" + selectedModel,
    })
      .then((response) => response.json())
      .then((data) => {
        predictionDiv.innerText = data.prediction;
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });
});
