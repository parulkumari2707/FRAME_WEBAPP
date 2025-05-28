// Script for FRAME Feature Selection Tool
/*
$(document).ready(function () {
    console.log("DOM is ready, initializing script...");

    // Handle file upload
    $('#upload-form').on('submit', function (e) {
        e.preventDefault();
        console.log("Upload form submitted");

        const fileInput = $('#file')[0];
        if (fileInput.files.length === 0) {
            showError('upload-error', 'Please select a CSV file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        console.log("File appended to form data:", fileInput.files[0].name);

        // Show loading indicator
        $('#upload-progress').removeClass('d-none');
        $('#upload-error').addClass('d-none');

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function (response) {
                console.log("Upload successful, response:", response);

                // Hide loading indicator
                $('#upload-progress').addClass('d-none');

                // Populate target column dropdown
                const targetSelect = $('#target-column');
                targetSelect.empty();
                targetSelect.append($('<option>', { value: '', text: 'Select target column' }));

                response.columns.forEach(function (column) {
                    targetSelect.append($('<option>', { value: column, text: column }));
                });

                // Set session ID
                $('#session-id').val(response.session_id);

                // Show target selection section
                $('#target-section').removeClass('d-none');

                // Scroll to target section
                $('html, body').animate({
                    scrollTop: $('#target-section').offset().top - 20
                }, 500);
            },
            error: function (xhr, status, error) {
                console.error("Upload error:", error);
                console.error("Status:", status);
                console.error("Response:", xhr.responseText);

                // Hide loading indicator
                $('#upload-progress').addClass('d-none');

                // Show error message
                let errorMsg = 'Failed to upload file. ' + error;
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMsg = xhr.responseJSON.error;
                }
                showError('upload-error', errorMsg);
            }
        });
    });

    // Handle target column selection and processing
    $('#target-form').on('submit', function (e) {
        e.preventDefault();
        console.log("Target form submitted");

        const targetColumn = $('#target-column').val();
        const sessionId = $('#session-id').val();
        const numFeatures = $('#num-features').val();
        const topK = $('#top-k').val();

        if (!targetColumn) {
            showError('process-error', 'Please select a target column.');
            return;
        }

        // Validate inputs
        if (parseInt(numFeatures) <= 0) {
            showError('process-error', 'Number of features must be greater than 0.');
            return;
        }

        if (parseInt(topK) <= 0) {
            showError('process-error', 'Top K must be greater than 0.');
            return;
        }

        if (parseInt(numFeatures) > parseInt(topK)) {
            showError('process-error', 'Number of features cannot be greater than Top K.');
            return;
        }

        // Show loading indicator
        $('#process-progress').removeClass('d-none');
        $('#process-error').addClass('d-none');

        console.log("Sending process request with data:", {
            session_id: sessionId,
            target_column: targetColumn,
            num_features: numFeatures,
            top_k: topK
        });

        $.ajax({
            url: '/process',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                session_id: sessionId,
                target_column: targetColumn,
                num_features: numFeatures,
                top_k: topK
            }),
            success: function (response) {
                console.log("Process successful, response:", response);

                // Hide loading indicator
                $('#process-progress').addClass('d-none');

                // Populate selected features list
                const featuresList = $('#selected-features-list');
                featuresList.empty();

                response.selected_features.forEach(function (feature) {
                    featuresList.append($('<li>', {
                        class: 'list-group-item',
                        text: feature
                    }));
                });

                // Store selected features for visualization
                window.selectedFeatures = response.selected_features;
                window.targetColumn = response.target_column;
                window.sessionId = response.session_id;
                window.isClassification = response.is_classification;

                // Show results section
                $('#results-section').removeClass('d-none');

                // Scroll to results section
                $('html, body').animate({
                    scrollTop: $('#results-section').offset().top - 20
                }, 500);
            },
            error: function (xhr, status, error) {
                console.error("Process error:", error);
                console.error("Status:", status);
                console.error("Response:", xhr.responseText);

                // Hide loading indicator
                $('#process-progress').addClass('d-none');

                // Show error message
                let errorMsg = 'Failed to process data. ' + error;
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMsg = xhr.responseJSON.error;
                }
                showError('process-error', errorMsg);
            }
        });
    });

    // Handle visualization button click
    $('#visualize-btn').on('click', function () {
        console.log("Visualize button clicked");

        if (!window.selectedFeatures || window.selectedFeatures.length === 0) {
            showError('visualization-error', 'No features selected for visualization.');
            return;
        }

        // Show loading indicator
        $('#visualization-progress').removeClass('d-none');
        $('#visualization-error').addClass('d-none');

        console.log("Sending visualization request with data:", {
            session_id: window.sessionId,
            target_column: window.targetColumn,
            selected_features: window.selectedFeatures,
            is_classification: window.isClassification
        });

        $.ajax({
            url: '/visualize',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                session_id: window.sessionId,
                target_column: window.targetColumn,
                selected_features: window.selectedFeatures,
                is_classification: window.isClassification
            }),
            success: function (response) {
                console.log("Visualization successful");

                // Hide loading indicator
                $('#visualization-progress').addClass('d-none');

                // Populate visualizations
                $('#importance-plot').attr('src', 'data:image/png;base64,' + response.importance_plot);
                $('#correlation-plot').attr('src', 'data:image/png;base64,' + response.correlation_plot);
                $('#performance-plot').attr('src', 'data:image/png;base64,' + response.performance_plot);

                // Display performance metrics
                const metricsHtml = `
                    <p><strong>${response.metric_name} with all features:</strong> ${response.metric_all.toFixed(4)}</p>
                    <p><strong>${response.metric_name} with selected features:</strong> ${response.metric_selected.toFixed(4)}</p>
                    <p class="text-${response.metric_selected >= response.metric_all ? 'success' : 'warning'}">
                        <strong>Improvement:</strong> 
                        ${((response.metric_selected - response.metric_all) * 100).toFixed(2)}%
                    </p>
                `;
                $('#performance-metrics').html(metricsHtml);

                // Show visualization section
                $('#viz-section').removeClass('d-none');

                // Scroll to visualization section
                $('html, body').animate({
                    scrollTop: $('#viz-section').offset().top - 20
                }, 500);
            },
            error: function (xhr, status, error) {
                console.error("Visualization error:", error);
                console.error("Status:", status);
                console.error("Response:", xhr.responseText);

                // Hide loading indicator
                $('#visualization-progress').addClass('d-none');

                // Show error message
                let errorMsg = 'Failed to generate visualizations. ' + error;
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMsg = xhr.responseJSON.error;
                }
                showError('visualization-error', errorMsg);
            }
        });
    });

    // Handle reset button click
    $('#reset-btn').on('click', function () {
        console.log("Reset button clicked");

        // Reset form
        $('#upload-form')[0].reset();
        $('#target-form')[0].reset();

        // Clear selected features
        $('#selected-features-list').empty();
        window.selectedFeatures = null;
        window.targetColumn = null;
        window.sessionId = null;

        // Hide sections
        $('#target-section').addClass('d-none');
        $('#results-section').addClass('d-none');
        $('#viz-section').addClass('d-none');

        // Show upload section
        $('#upload-section').removeClass('d-none');

        // Scroll to top
        $('html, body').animate({
            scrollTop: $('#upload-section').offset().top - 20
        }, 500);
    });

    // Helper function to show error messages
    function showError(elementId, message) {
        console.error("Showing error in", elementId, ":", message);
        const errorElement = $('#' + elementId);
        errorElement.text(message);
        errorElement.removeClass('d-none');
    }
});
*/
// Script for FRAME Feature Selection Tool
$(document).ready(function () {
    console.log("DOM is ready, initializing script...");

    // Handle file upload
    $('#upload-form').on('submit', function (e) {
        e.preventDefault();
        console.log("Upload form submitted");

        const fileInput = $('#file')[0];
        if (fileInput.files.length === 0) {
            showError('upload-error', 'Please select a CSV file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        console.log("File appended to form data:", fileInput.files[0].name);

        // Show loading indicator
        $('#upload-progress').removeClass('d-none');
        $('#upload-error').addClass('d-none');

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function (response) {
                console.log("Upload successful, response:", response);

                // Hide loading indicator
                $('#upload-progress').addClass('d-none');

                // Populate target column dropdown
                const targetSelect = $('#target-column');
                targetSelect.empty();
                targetSelect.append($('<option>', { value: '', text: 'Select target column' }));

                response.columns.forEach(function (column) {
                    targetSelect.append($('<option>', { value: column, text: column }));
                });

                // Set session ID
                $('#session-id').val(response.session_id);

                // Show target selection section
                $('#target-section').removeClass('d-none');

                // Scroll to target section
                $('html, body').animate({
                    scrollTop: $('#target-section').offset().top - 20
                }, 500);
            },
            error: function (xhr, status, error) {
                console.error("Upload error:", error);
                console.error("Status:", status);
                console.error("Response:", xhr.responseText);

                // Hide loading indicator
                $('#upload-progress').addClass('d-none');

                // Show error message
                let errorMsg = 'Failed to upload file. ' + error;
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMsg = xhr.responseJSON.error;
                }
                showError('upload-error', errorMsg);
            }
        });
    });

    // Handle target column selection and processing
    $('#target-form').on('submit', function (e) {
        e.preventDefault();
        console.log("Target form submitted");

        const targetColumn = $('#target-column').val();
        const sessionId = $('#session-id').val();
        const numFeatures = $('#num-features').val();
        const topK = $('#top-k').val();

        if (!targetColumn) {
            showError('process-error', 'Please select a target column.');
            return;
        }

        if (parseInt(numFeatures) <= 0) {
            showError('process-error', 'Number of features must be greater than 0.');
            return;
        }

        if (parseInt(topK) <= 0) {
            showError('process-error', 'Top K must be greater than 0.');
            return;
        }

        if (parseInt(numFeatures) > parseInt(topK)) {
            showError('process-error', 'Number of features cannot be greater than Top K.');
            return;
        }

        $('#process-progress').removeClass('d-none');
        $('#process-error').addClass('d-none');

        console.log("Sending process request with data:", {
            session_id: sessionId,
            target_column: targetColumn,
            num_features: numFeatures,
            top_k: topK
        });

        $.ajax({
            url: '/process',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                session_id: sessionId,
                target_column: targetColumn,
                num_features: numFeatures,
                top_k: topK
            }),
            success: function (response) {
                console.log("Process successful, response:", response);

                $('#process-progress').addClass('d-none');

                const featuresList = $('#selected-features-list');
                featuresList.empty();

                response.selected_features.forEach(function (feature) {
                    featuresList.append($('<li>', {
                        class: 'list-group-item',
                        text: feature
                    }));
                });

                window.selectedFeatures = response.selected_features;
                window.targetColumn = response.target_column;
                window.sessionId = response.session_id;
                window.isClassification = response.is_classification;

                $('#results-section').removeClass('d-none');

                $('html, body').animate({
                    scrollTop: $('#results-section').offset().top - 20
                }, 500);
            },
            error: function (xhr, status, error) {
                console.error("Process error:", error);
                console.error("Status:", status);
                console.error("Response:", xhr.responseText);

                $('#process-progress').addClass('d-none');

                let errorMsg = 'Failed to process data. ' + error;
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMsg = xhr.responseJSON.error;
                }
                showError('process-error', errorMsg);
            }
        });
    });

    // Handle visualization button click
    $('#visualize-btn').on('click', function () {
        console.log("Visualize button clicked");

        if (!window.selectedFeatures || window.selectedFeatures.length === 0) {
            showError('visualization-error', 'No features selected for visualization.');
            return;
        }

        $('#visualization-progress').removeClass('d-none');
        $('#visualization-error').addClass('d-none');

        console.log("Sending visualization request with data:", {
            session_id: window.sessionId,
            target_column: window.targetColumn,
            selected_features: window.selectedFeatures,
            is_classification: window.isClassification
        });

        $.ajax({
            url: '/visualize',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                session_id: window.sessionId,
                target_column: window.targetColumn,
                selected_features: window.selectedFeatures,
                is_classification: window.isClassification
            }),
            success: function (response) {
                console.log("Visualization successful");

                $('#visualization-progress').addClass('d-none');

                $('#importance-plot').attr('src', 'data:image/png;base64,' + response.importance_plot);
                $('#correlation-plot').attr('src', 'data:image/png;base64,' + response.correlation_plot);
                $('#performance-plot').attr('src', 'data:image/png;base64,' + response.performance_plot);

                let metricsHtml = `
                    <p><strong>${response.metric_name} with all features:</strong> ${response.metric_all.toFixed(4)}</p>
                    <p><strong>${response.metric_name} with selected features:</strong> ${response.metric_selected.toFixed(4)}</p>
                    <p class="text-${response.metric_selected >= response.metric_all ? 'success' : 'warning'}">
                        <strong>Improvement:</strong> 
                        ${((response.metric_selected - response.metric_all) * 100).toFixed(2)}%
                    </p>
                `;

                if (response.hasOwnProperty('rmse_all')) {
                    metricsHtml += `
                        <hr>
                        <p><strong>RMSE with all features:</strong> ${response.rmse_all.toFixed(4)}</p>
                        <p><strong>RMSE with selected features:</strong> ${response.rmse_selected.toFixed(4)}</p>
                        <p class="text-${response.rmse_selected <= response.rmse_all ? 'success' : 'warning'}">
                            <strong>Improvement:</strong> 
                            ${((response.rmse_all - response.rmse_selected) / response.rmse_all * 100).toFixed(2)}%
                        </p>
                    `;
                }

                $('#performance-metrics').html(metricsHtml);

                $('#viz-section').removeClass('d-none');

                $('html, body').animate({
                    scrollTop: $('#viz-section').offset().top - 20
                }, 500);
            },
            error: function (xhr, status, error) {
                console.error("Visualization error:", error);
                console.error("Status:", status);
                console.error("Response:", xhr.responseText);

                $('#visualization-progress').addClass('d-none');

                let errorMsg = 'Failed to generate visualizations. ' + error;
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMsg = xhr.responseJSON.error;
                }
                showError('visualization-error', errorMsg);
            }
        });
    });

    // Handle reset button click
    $('#reset-btn').on('click', function () {
        console.log("Reset button clicked");

        $('#upload-form')[0].reset();
        $('#target-form')[0].reset();

        $('#selected-features-list').empty();
        window.selectedFeatures = null;
        window.targetColumn = null;
        window.sessionId = null;

        $('#target-section').addClass('d-none');
        $('#results-section').addClass('d-none');
        $('#viz-section').addClass('d-none');

        $('#upload-section').removeClass('d-none');

        $('html, body').animate({
            scrollTop: $('#upload-section').offset().top - 20
        }, 500);
    });

    // Helper function to show error messages
    function showError(elementId, message) {
        console.error("Showing error in", elementId, ":", message);
        const errorElement = $('#' + elementId);
        errorElement.text(message);
        errorElement.removeClass('d-none');
    }
});
