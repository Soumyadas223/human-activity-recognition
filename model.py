"""
CNN-LSTM Model Architecture for Human Activity Recognition
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ActivityRecognitionCNNLSTM(nn.Module):
    """
    CNN-LSTM model for video activity recognition
    
    Architecture:
    - CNN: ResNet18 (pretrained) for spatial feature extraction
    - LSTM: Bidirectional LSTM for temporal modeling
    - FC layers: Classification head
    """
    
    def __init__(self, num_classes=6, num_frames=16, hidden_dim=256, num_layers=2, dropout=0.5):
        """
        Initialize the model
        
        Args:
            num_classes: Number of activity classes
            num_frames: Number of frames in sequence
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(ActivityRecognitionCNNLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # Remove the last fully connected layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # CNN output dimension (ResNet18 outputs 512 features)
        self.cnn_output_dim = 512
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # LSTM output dimension (bidirectional)
        self.lstm_output_dim = hidden_dim * 2
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(self.lstm_output_dim, self.lstm_output_dim),
            nn.Tanh(),
            nn.Linear(self.lstm_output_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.lstm_output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for new layers"""
        for module in [self.fc1, self.fc2]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_frames, 3, height, width)
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        batch_size = x.size(0)
        num_frames = x.size(1)
        
        # Reshape to (batch_size * num_frames, 3, height, width)
        x = x.view(batch_size * num_frames, x.size(2), x.size(3), x.size(4))
        
        # Extract spatial features using CNN
        x = self.cnn(x)  # (batch_size * num_frames, 512, 1, 1)
        x = x.view(batch_size * num_frames, self.cnn_output_dim)  # (batch_size * num_frames, 512)
        
        # Reshape back to (batch_size, num_frames, 512)
        x = x.view(batch_size, num_frames, self.cnn_output_dim)
        
        # LSTM for temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, num_frames, lstm_output_dim)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch_size, num_frames, 1)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, lstm_output_dim)
        
        # Classification
        x = self.dropout(attended_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class ActivityRecognitionCNN3D(nn.Module):
    """
    3D CNN model for video activity recognition
    
    Uses 3D convolutions to capture both spatial and temporal features directly
    """
    
    def __init__(self, num_classes=6, dropout=0.5):
        """
        Initialize 3D CNN model
        
        Args:
            num_classes: Number of activity classes
            dropout: Dropout rate
        """
        super(ActivityRecognitionCNN3D, self).__init__()
        
        self.num_classes = num_classes
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Adaptive pooling and classification
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_frames, 3, height, width)
                or (batch_size, 3, num_frames, height, width)
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # Convert to (batch_size, 3, num_frames, height, width) if needed
        if x.size(1) == 3:
            pass  # Already in correct format
        else:
            # Assume (batch_size, num_frames, 3, height, width)
            x = x.permute(0, 2, 1, 3, 4)  # -> (batch_size, 3, num_frames, height, width)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool4(x)
        
        # Classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ActivityRecognitionSimpleLSTM(nn.Module):
    """
    Simple LSTM model for activity recognition
    Works directly on flattened frame features
    """
    
    def __init__(self, num_classes=6, input_size=512, hidden_dim=256, num_layers=2, dropout=0.5):
        """
        Initialize Simple LSTM model
        
        Args:
            num_classes: Number of activity classes
            input_size: Size of input features per frame
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(ActivityRecognitionSimpleLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_frames, input_size)
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Classification
        x = self.dropout(last_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


def create_model(model_type='cnnlstm', num_classes=6, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('cnnlstm', 'cnn3d', 'simplelstm')
        num_classes: Number of classes
        **kwargs: Additional arguments for model initialization
    
    Returns:
        model: Initialized model
    """
    if model_type == 'cnnlstm':
        return ActivityRecognitionCNNLSTM(num_classes=num_classes, **kwargs)
    elif model_type == 'cnn3d':
        return ActivityRecognitionCNN3D(num_classes=num_classes, **kwargs)
    elif model_type == 'simplelstm':
        return ActivityRecognitionSimpleLSTM(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
