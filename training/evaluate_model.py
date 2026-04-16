import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model_path="models/weights/deepfake_model_final.h5"):
        self.model = tf.keras.models.load_model(model_path)
        
    def evaluate_on_test_set(self, test_generator):
        """تقييم شامل على مجموعة الاختبار"""
        
        print("=" * 60)
        print("📊 MODEL EVALUATION")
        print("=" * 60)
        
        # توقع النتائج
        predictions = self.model.predict(test_generator)
        predictions_binary = (predictions > 0.5).astype(int)
        
        # الحصول على true labels
        true_labels = test_generator.classes
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        # Classification Report
        print("\n📋 Classification Report:")
        print(classification_report(true_labels, predictions_binary,
                                   target_names=['Real', 'Fake']))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.show()
        
        # حساب مقاييس إضافية
        accuracy = np.mean(predictions_binary.flatten() == true_labels)
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n📈 Summary Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': roc_auc,
            'confusion_matrix': cm
        }

if __name__ == "__main__":
    # هذا يتطلب وجود test_generator
    print("Run train_deepfake_model.py first to create the model")