// Configuration
const API_CHAT_URL = "http://127.0.0.1:5000/chat"; // Ensure Flask server is running

// Elements
const chatDisplay = document.getElementById('chatDisplay');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

// Sidebar toggle elements
const sidebarToggle = document.getElementById('sidebarToggle'); // Button to open sidebar
const closeSidebarButton = document.getElementById('closeSidebar'); // Button to close sidebar
const sidebar = document.getElementById('sidebar');

// File input and attach button elements
const attachButton = document.getElementById('attachButton');
const fileInput = document.getElementById('fileInput');

// Image Name Display Elements
const imageNameBox = document.getElementById('imageNameBox');
const imageNameSpan = document.getElementById('imageName');
const removeImageButton = document.getElementById('removeImageButton');

// Variable for the "Diagnosing..." animation
let diagnosingInterval;

// Variable to store the attached file
let attachedFile = null;
let attachedFileDataURL = null; // Variable to store the Data URL of the attached image

// Initial Chat State
let messages = [
    { role: "bot", content: "Hello! How can I assist you today?" }
];

// Function to render messages
function renderMessages() {
    chatDisplay.innerHTML = ''; // Clear current messages
    messages.forEach(message => {
        if (message.role === "user") {
            // User Message
            const userContainer = document.createElement('div');
            userContainer.className = 'chat-container user-chat-container';

            const chatContent = document.createElement('div');
            chatContent.className = 'chat-content user-bubble chat-bubble';

            let messageContent = `<span class="sender-label">You:</span> <span>${sanitize(message.content)}</span>`;

            if (message.imageData) {
                messageContent += `<br><img src="${message.imageData}" alt="Attached Image" class="image-attachment">`;
            }

            chatContent.innerHTML = messageContent;

            const userIcon = document.createElement('img');
            userIcon.src = 'images/user.png'; // Ensure this image exists
            userIcon.alt = 'User';
            userIcon.className = 'message-icon';

            userContainer.appendChild(chatContent);
            userContainer.appendChild(userIcon);
            chatDisplay.appendChild(userContainer);
        } else {
            // Bot Message
            const botContainer = document.createElement('div');
            botContainer.className = 'chat-container bot-chat-container';

            const botIcon = document.createElement('img');
            botIcon.src = 'images/logo-1-removebg.png'; // Ensure this image exists
            botIcon.alt = 'Doctor';
            botIcon.className = 'message-icon';

            const chatContent = document.createElement('div');
            chatContent.className = 'chat-content bot-bubble chat-bubble';
            chatContent.innerHTML = `<span class="sender-label">Doctor:</span> <span>${sanitize(message.content)}</span>`;

            botContainer.appendChild(botIcon);
            botContainer.appendChild(chatContent);
            chatDisplay.appendChild(botContainer);
        }
    });

    // Automatically scroll to the bottom after rendering messages
    scrollToBottom();
}

// Function to sanitize user input to prevent XSS
function sanitize(str) {
    const temp = document.createElement('div');
    temp.textContent = str;
    return temp.innerHTML;
}

// Function to send user message
function sendMessage() {
    const userText = userInput.value.trim();
    if (userText === "" && !attachedFile) return;

    // Append user message
    messages.push({ role: "user", content: userText, imageData: attachedFileDataURL });
    renderMessages();

    // Clear input
    userInput.value = "";

    // Hide the image name box since the image is being sent
    hideImageNameBox();

    // Show "Diagnosing..." animation
    showDiagnosingAnimation();

    // Show upload indicator if file is attached
    if (attachedFile) {
        showUploadIndicator(); // This will now include a progress bar
    }

    // Create FormData
    let formData = new FormData();
    formData.append('message', userText);
    if (attachedFile) {
        formData.append('image', attachedFile);
    }

    // Initialize XMLHttpRequest
    const xhr = new XMLHttpRequest();

    xhr.open('POST', API_CHAT_URL, true);

    // Set up progress event listener
    xhr.upload.addEventListener('progress', function(event) {
        if (event.lengthComputable) {
            const percentComplete = (event.loaded / event.total) * 100;
            updateProgressBar(percentComplete);
        }
    });

    // Set up onload handler
    xhr.onload = function() {
        if (xhr.status === 200) {
            const data = JSON.parse(xhr.responseText);
            const gptResponse = data.gpt_response || "No response from the chatbot.";

            // Stop the "Diagnosing..." animation
            stopDiagnosingAnimation();

            // Remove the "Diagnosing..." message
            messages.pop();

            // Append the actual response
            messages.push({ role: "bot", content: gptResponse });
        } else {
            // Handle non-200 responses
            stopDiagnosingAnimation();
            messages.pop();
            messages.push({ role: "bot", content: "An error occurred while processing your request. Please try again later." });
        }

        if (attachedFile) {
            hideUploadIndicator();
            attachedFile = null;
            attachedFileDataURL = null;
            fileInput.value = '';
            removeFileAttachedIndicator();
        }

        renderMessages();
    };

    // Set up onerror handler
    xhr.onerror = function() {
        // Handle network errors
        console.error('An error occurred during the transaction');
        stopDiagnosingAnimation();
        messages.pop();
        messages.push({ role: "bot", content: "An error occurred while sending your message. Please try again." });

        if (attachedFile) {
            hideUploadIndicator();
            attachedFile = null;
            attachedFileDataURL = null;
            fileInput.value = '';
            removeFileAttachedIndicator();
        }

        renderMessages();
    };

    // Send the request
    xhr.send(formData);
}

// Function to start "Diagnosing..." live animation
function showDiagnosingAnimation() {
    let dots = 0;
    const diagnosingMessage = { role: "bot", content: "Diagnosing..." };
    messages.push(diagnosingMessage);

    // Set up an interval to update the "Diagnosing..." message
    diagnosingInterval = setInterval(() => {
        dots = (dots + 1) % 4; // Cycle between 0, 1, 2, 3 dots
        diagnosingMessage.content = "Diagnosing" + ".".repeat(dots);
        renderMessages();
    }, 500); // Update every half second
}

// Function to stop the "Diagnosing..." animation
function stopDiagnosingAnimation() {
    clearInterval(diagnosingInterval); // Stop the interval
}

// Event listener for send button
sendButton.addEventListener('click', sendMessage);

// Allow sending message by pressing Enter key
userInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
});

// Event listener for attach button
attachButton.addEventListener('click', function () {
    fileInput.click(); // Trigger the file input to open the file explorer
});

// Handle file selection
fileInput.addEventListener('change', function () {
    if (fileInput.files.length > 0) {
        attachedFile = fileInput.files[0];
        showFileAttachedIndicator(attachedFile.name);

        // Show the image name box with the file name
        showImageNameBox(attachedFile.name);

        // Read the file as a data URL and store it
        const reader = new FileReader();
        reader.onload = function(e) {
            attachedFileDataURL = e.target.result; // Store the data URL
        };
        reader.readAsDataURL(attachedFile);
    }
});

// Show file attached indicator
function showFileAttachedIndicator(filename) {
    attachButton.classList.add('file-attached');
    attachButton.title = `File attached: ${filename}`;
}

// Remove file attached indicator
function removeFileAttachedIndicator() {
    attachButton.classList.remove('file-attached');
    attachButton.title = 'Attach File';
}

// Show image name box
function showImageNameBox(filename) {
    imageNameSpan.textContent = `Attached: ${filename}`;
    imageNameBox.style.display = 'flex';
}

// Hide image name box
function hideImageNameBox() {
    imageNameBox.style.display = 'none';
    imageNameSpan.textContent = '';
    attachedFile = null;
    attachedFileDataURL = null;
    fileInput.value = '';
    removeFileAttachedIndicator();
}

// Show upload indicator
function showUploadIndicator() {
    let uploadIndicator = document.getElementById('uploadIndicator');
    if (!uploadIndicator) {
        uploadIndicator = document.createElement('div');
        uploadIndicator.id = 'uploadIndicator';
        uploadIndicator.className = 'upload-indicator';
        
        // Create progress bar container
        const progressContainer = document.createElement('div');
        progressContainer.className = 'progress-container';
        
        // Create progress bar
        const progressBar = document.createElement('div');
        progressBar.id = 'uploadProgress';
        progressBar.className = 'progress-bar';
        progressBar.textContent = '0%';
        
        progressContainer.appendChild(progressBar);
        uploadIndicator.appendChild(progressContainer);

        document.body.appendChild(uploadIndicator);
    }
    uploadIndicator.style.display = 'block';
}

// Hide upload indicator
function hideUploadIndicator() {
    let uploadIndicator = document.getElementById('uploadIndicator');
    if (uploadIndicator) {
        uploadIndicator.style.display = 'none';
    }
}

// Function to update the progress bar
function updateProgressBar(percent) {
    const progressBar = document.getElementById('uploadProgress');
    if (progressBar) {
        progressBar.style.width = percent + '%';
        progressBar.textContent = Math.floor(percent) + '%';
    }
}

// Event listener for remove image button
removeImageButton.addEventListener('click', function() {
    hideImageNameBox();
});

// Initial render
renderMessages();

// Function to toggle sidebar and button visibility
function toggleSidebar() {
    sidebar.classList.toggle('closed');
    
    // Show/hide sidebar toggle buttons based on sidebar state
    if (sidebar.classList.contains('closed')) {
        sidebarToggle.style.display = 'block'; // Show open button
        closeSidebarButton.style.display = 'none'; // Hide close button
    } else {
        sidebarToggle.style.display = 'none'; // Hide open button
        closeSidebarButton.style.display = 'block'; // Show close button
    }
}

// Event listeners for open/close sidebar buttons
sidebarToggle.addEventListener('click', toggleSidebar);
closeSidebarButton.addEventListener('click', toggleSidebar);

// Hide the sidebar toggle initially (when sidebar is open)
sidebarToggle.style.display = 'none';

// Scroll the chat to the bottom
function scrollToBottom() {
    chatDisplay.scrollTop = chatDisplay.scrollHeight; 
}
