def create_data_augmentation_functions():
    """
    Create a set of data augmentation functions for sign language landmark sequences
    """
    import numpy as np
    
    def random_temporal_shift(landmarks, max_shift=5):
        """Randomly shift landmarks in time dimension"""
        if np.random.random() < 0.5:
            return landmarks
            
        frames, num_landmarks, coords = landmarks.shape
        shift = np.random.randint(1, max_shift)
        direction = np.random.choice([-1, 1])
        
        if direction > 0:  # Shift right
            shifted = np.zeros_like(landmarks)
            shifted[shift:, :, :] = landmarks[:-shift, :, :]
        else:  # Shift left
            shifted = np.zeros_like(landmarks)
            shifted[:-shift, :, :] = landmarks[shift:, :, :]
            
        return shifted
    
    def random_spatial_jitter(landmarks, jitter_range=0.02):
        """Add random jitter to landmark positions"""
        if np.random.random() < 0.5:
            return landmarks
            
        jitter = np.random.uniform(-jitter_range, jitter_range, landmarks.shape)
        return landmarks + jitter
    
    def random_scale(landmarks, scale_range=(0.9, 1.1)):
        """Randomly scale landmark positions"""
        if np.random.random() < 0.5:
            return landmarks
            
        scale = np.random.uniform(*scale_range)
        
        # Calculate center of landmarks (average position)
        center = np.mean(landmarks, axis=(0, 1))
        
        # Scale landmarks around center
        scaled = center + scale * (landmarks - center)
        return scaled
    
    def random_rotation(landmarks, max_angle=15):
        """Randomly rotate landmarks in 3D space (around y-axis)"""
        if np.random.random() < 0.5:
            return landmarks
            
        angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
        
        # Create rotation matrix around y-axis
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        
        # Calculate center of landmarks
        center = np.mean(landmarks, axis=(0, 1))
        
        # Rotate landmarks around center
        centered = landmarks - center
        rotated = np.zeros_like(landmarks)
        
        for i in range(landmarks.shape[0]):
            for j in range(landmarks.shape[1]):
                rotated[i, j] = np.dot(rotation_matrix, centered[i, j])
                
        rotated = rotated + center
        return rotated
    
    def random_dropout(landmarks, drop_prob=0.05, min_landmarks=10):
        """Randomly drop some landmarks to simulate occlusion"""
        if np.random.random() < 0.5:
            return landmarks
            
        frames, num_landmarks, coords = landmarks.shape
        
        # Determine how many landmarks to keep (at least min_landmarks)
        keep_n = max(min_landmarks, int(num_landmarks * (1 - drop_prob)))
        
        # Randomly select landmarks to keep
        keep_indices = np.random.choice(num_landmarks, keep_n, replace=False)
        
        # Create mask for landmarks
        mask = np.zeros(num_landmarks, dtype=bool)
        mask[keep_indices] = True
        
        # Apply mask (set dropped landmarks to zero)
        result = landmarks.copy()
        result[:, ~mask, :] = 0
        
        return result
    
    def random_time_warp(landmarks, warp_factor=0.2):
        """Apply random temporal warping"""
        if np.random.random() < 0.5:
            return landmarks
            
        frames, num_landmarks, coords = landmarks.shape
        
        # Create random warping function
        source_frames = np.arange(frames)
        target_frames = np.arange(frames)
        
        # Add random displacement to target frames
        displacement = np.random.uniform(-warp_factor, warp_factor, frames)
        smoothed_displacement = np.convolve(displacement, np.ones(5)/5, mode='same')
        target_frames = target_frames + smoothed_displacement * frames
        
        # Ensure target frames are within bounds
        target_frames = np.clip(target_frames, 0, frames-1)
        
        # Create warped sequence
        warped = np.zeros_like(landmarks)
        
        for i, target_frame in enumerate(target_frames):
            # Find closest original frames
            low_idx = int(np.floor(target_frame))
            high_idx = int(np.ceil(target_frame))
            
            if low_idx == high_idx:
                warped[i] = landmarks[low_idx]
            else:
                # Interpolate between frames
                weight = target_frame - low_idx
                warped[i] = (1 - weight) * landmarks[low_idx] + weight * landmarks[high_idx]
        
        return warped
    
    def random_speed(landmarks, speed_range=(0.8, 1.2)):
        """Change the speed of the sequence"""
        if np.random.random() < 0.5:
            return landmarks
            
        frames, num_landmarks, coords = landmarks.shape
        speed_factor = np.random.uniform(*speed_range)
        
        # Calculate new number of frames
        new_frames = int(frames / speed_factor)
        if new_frames <= 1 or new_frames >= frames * 2:
            return landmarks
            
        # Create indices for interpolation
        orig_indices = np.arange(frames)
        new_indices = np.linspace(0, frames-1, new_frames)
        
        # Create new sequence
        new_sequence = np.zeros((new_frames, num_landmarks, coords))
        
        # Interpolate for each landmark and dimension
        for l in range(num_landmarks):
            for c in range(coords):
                new_sequence[:, l, c] = np.interp(new_indices, orig_indices, landmarks[:, l, c])
        
        # If new sequence is shorter, pad with zeros
        if new_frames < frames:
            padded = np.zeros_like(landmarks)
            padded[:new_frames] = new_sequence
            return padded
        
        # If new sequence is longer, truncate
        return new_sequence[:frames]
    
    def normalize_landmarks(landmarks):
        """Normalize landmarks to a standard scale and position"""
        # Find min and max values across all dimensions
        min_vals = np.min(landmarks, axis=(0, 1))
        max_vals = np.max(landmarks, axis=(0, 1))
        
        # Scale to range [0, 1]
        range_vals = max_vals - min_vals
        # Avoid division by zero
        range_vals[range_vals == 0] = 1
        
        normalized = (landmarks - min_vals) / range_vals
        
        return normalized
    
    def augment_landmarks(landmarks, augment_prob=0.8):
        """Apply a sequence of random augmentations"""
        if np.random.random() > augment_prob:
            return landmarks
        
        # Make a copy to avoid modifying the original
        aug_landmarks = landmarks.copy()
        
        # Apply sequence of augmentations randomly
        augmentation_functions = [
            random_temporal_shift,
            random_spatial_jitter,
            random_scale,
            random_rotation,
            random_dropout,
            random_time_warp,
            random_speed
        ]
        
        # Shuffle the functions to apply them in random order
        np.random.shuffle(augmentation_functions)
        
        # Apply 1-3 random augmentations
        num_augs = np.random.randint(1, 4)
        
        for i in range(min(num_augs, len(augmentation_functions))):
            aug_landmarks = augmentation_functions[i](aug_landmarks)
        
        # Always normalize at the end
        aug_landmarks = normalize_landmarks(aug_landmarks)
        
        return aug_landmarks
    
    return {
        'temporal_shift': random_temporal_shift,
        'spatial_jitter': random_spatial_jitter,
        'scale': random_scale,
        'rotation': random_rotation,
        'dropout': random_dropout,
        'time_warp': random_time_warp,
        'speed': random_speed,
        'normalize': normalize_landmarks,
        'augment': augment_landmarks
    }
