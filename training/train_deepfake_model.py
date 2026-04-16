import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
callbacks = keras.callbacks
import numpy as np
import os
import matplotlib.pyplot as plt

class DeepfakeTrainer:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        
    def create_data_generators(self):
        """إنشاء مولدات البيانات"""
        
        # Data augmentation للتدريب
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # بدون augmentation للتحقق والاختبار
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        # تحميل البيانات
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake']
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake']
        )
        
        test_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake']
        )
        
        return train_generator, val_generator, test_generator
    
    def build_efficientnet_model(self):
        """بناء نموذج EfficientNetB0"""
        
        # Load pretrained EfficientNet
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # تجميد الطبقات الأولى
        base_model.trainable = False
        
        # إضافة طبقات مخصصة
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        return model, base_model
    
    def train(self):
        """تنفيذ التدريب"""
        
        print("=" * 60)
        print("🚀 STARTING DEEPFAKE MODEL TRAINING")
        print("=" * 60)
        
        # 1. تحميل البيانات
        print("\n📊 Loading data...")
        train_gen, val_gen, test_gen = self.create_data_generators()
        
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        
        # 2. بناء النموذج
        print("\n🏗️ Building model...")
        model, base_model = self.build_efficientnet_model()
        
        # 3. Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # 4. Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'models/weights/deepfake_model_phase1.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 5. المرحلة 1: تدريب الطبقات المضافة فقط
        print("\n📚 PHASE 1: Training top layers...")
        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # 6. المرحلة 2: Fine-tuning
        print("\n🔓 PHASE 2: Fine-tuning...")
        base_model.trainable = True
        
        # تجميد أول 100 طبقة
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # إعادة compile بمعدل تعلم أقل
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=30,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # 7. حفظ النموذج النهائي
        model.save('models/weights/deepfake_model_final.h5')
        print("\n✅ Model saved to models/weights/deepfake_model_final.h5")
        
        # 8. تقييم النموذج
        print("\n📊 Evaluating on test set...")
        test_loss, test_acc, test_auc = model.evaluate(test_gen, verbose=1)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # 9. رسم النتائج
        self.plot_training_history(history1, history2)
        
        return model, history1, history2
    
    def plot_training_history(self, history1, history2):
        """رسم منحنيات التدريب"""
        
        # دمج التاريخين
        hist = {}
        for key in history1.history.keys():
            hist[key] = history1.history[key] + history2.history[key]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(hist['accuracy'], label='Train Accuracy')
        axes[0].plot(hist['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Loss
        axes[1].plot(hist['loss'], label='Train Loss')
        axes[1].plot(hist['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

if __name__ == "__main__":
    trainer = DeepfakeTrainer()
    model, hist1, hist2 = trainer.train()