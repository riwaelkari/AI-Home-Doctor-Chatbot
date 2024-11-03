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

// Function to send user message using Fetch API
async function sendMessage() {
    console.log('sendMessage function called'); // Debugging statement

    const userText = userInput.value.trim();
    console.log(`User input: "${userText}"`); // Debugging

    if (userText === "" && !attachedFile) {
        console.log('No message or file to send'); // Debugging
        return;
    }
    if(attachedFile){
         console.log('test1')
    }

    // Append user message to chat display
    messages.push({ role: "user", content: userText, imageData: attachedFileDataURL });
    renderMessages();

    if(attachedFile){
        console.log('test2')
   }
    // Clear input
    userInput.value = "";
    if(attachedFile){
        console.log('test3')
   }
    // Hide the image name box since the image is being sent
    hideImageNameBox();
    if(attachedFile){
        console.log('test4')
   }
    // Show "Diagnosing..." animation
    showDiagnosingAnimation();
    if(attachedFile){
        console.log('test1')
   }
    // Show upload indicator if file is attached
    if (attachedFile) {
        showUploadIndicator(); // This will now include a progress bar
    }

    // Create FormData
    let formData = new FormData();
    formData.append('message', userText);
    if (attachedFile) {
        formData.append('image', attachedFile, attachedFile.name); // Include filename
        console.log(`Appending image: ${attachedFile.name}`); // Debugging
    }

    // Debugging: Log FormData contents
    console.log('FormData contents:');
    for (let pair of formData.entries()) {
        if (pair[0] === 'image') {
            console.log(`${pair[0]}: ${pair[1].name}`); // Log image filename
        } else {
            console.log(`${pair[0]}: ${pair[1]}`);
        }
    }

    try {
        const response = await fetch(API_CHAT_URL, {
            method: 'POST',
            body: formData
            // Note: Do NOT set the 'Content-Type' header when sending FormData
            // The browser will automatically set it, including the boundary
        });

        if (response.ok) {
            const data = await response.json();
            const gptResponse = data.gpt_response || "No response from the chatbot.";

            // Stop the "Diagnosing..." animation
            stopDiagnosingAnimation();

            // Remove the "Diagnosing..." message
            messages.pop();

            // Append the actual response
            messages.push({ role: "bot", content: gptResponse });
        } else {
            // Handle non-200 responses
            console.error(`Server responded with status ${response.status}`);
            const errorData = await response.json();
            const errorMessage = errorData.error || "An error occurred while processing your request.";
            stopDiagnosingAnimation();
            messages.pop();
            messages.push({ role: "bot", content: `Error: ${errorMessage}` });
        }

        if (attachedFile) {
            hideUploadIndicator();
            attachedFile = null;
            attachedFileDataURL = null;
            fileInput.value = '';
            removeFileAttachedIndicator();
        }

        renderMessages();
    } catch (error) {
        // Handle network or other errors
        console.error('An error occurred during the transaction:', error);
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
    }
}

// Function to start "Diagnosing..." live animation
function showDiagnosingAnimation() {
    let dots = 0;
    const diagnosingMessage = { role: "bot", content: "Diagnosing..." };
    messages.push(diagnosingMessage);
    renderMessages();

    // Set up an interval to update the "Diagnosing..." message
    diagnosingInterval = setInterval(() => {
        dots = (dots + 1) % 4; // Cycle between 0, 1, 2, 3 dots
        diagnosingMessage.content = "Diagnosing" + ".".repeat(dots);
        renderMessages();
    }, 500); // Update every half second
}

// Function to stop the "Diagnosing..." animation
function stopDiagnosingAnimation() {
    if (diagnosingInterval) {
        clearInterval(diagnosingInterval); // Stop the interval
        diagnosingInterval = null;
    }
}

// Event listener for send button
sendButton.addEventListener('click', function() {
    console.log('Send button clicked'); // Debugging
    sendMessage();
});

// Allow sending message by pressing Enter key
userInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        console.log('Enter key pressed in input'); // Debugging
        sendMessage();
    }
});

// Event listener for attach button
attachButton.addEventListener('click', function () {
    console.log('Attach button clicked'); // Debugging
    fileInput.click(); // Trigger the file input to open the file explorer
});

// Handle file selection
fileInput.addEventListener('change', function () {
    if (fileInput.files.length > 0) {
        attachedFile = fileInput.files[0];
        console.log(`File selected: ${attachedFile.name}`); // Debugging
        showFileAttachedIndicator(attachedFile.name);

        // Show the image name box with the file name
        showImageNameBox(attachedFile.name);

        // Read the file as a data URL and store it
        const reader = new FileReader();
        reader.onload = function(e) {
            attachedFileDataURL = e.target.result; // Store the data URL
            console.log(`File read as Data URL: ${attachedFileDataURL.substring(0, 30)}...`); // Debugging
        };
        reader.readAsDataURL(attachedFile);
    } else {
        console.log('No file selected'); // Debugging
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
    console.log('Remove image button clicked'); // Debugging
    hideImageNameBox();
});

// Initial render
renderMessages();

// Function to toggle sidebar and button visibility
function toggleSidebar() {
    sidebar.classList.toggle('closed');
    console.log(`Sidebar toggled. Closed: ${sidebar.classList.contains('closed')}`); // Debugging
    
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
