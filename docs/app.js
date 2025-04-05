// 1. 加载 TensorFlow.js
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js';

// 2. 模型和DOM元素引用
let model;
const uploadBtn = document.getElementById('upload');
const previewImg = document.getElementById('preview');
const resultDiv = document.getElementById('result');

// 3. 加载模型
async function loadModel() {
    showLoading("模型加载中...");
    try {
        model = await tf.loadLayersModel('models/your_model/model.json');
        console.log("模型加载成功");
    } catch (err) {
        showError(`模型加载失败: ${err.message}`);
    } finally {
        hideLoading();
    }
}

// 4. 图片预处理
function preprocessImage(imgElement) {
    return tf.tidy(() => {
        return tf.browser.fromPixels(imgElement)
            .resizeNearestNeighbor([224, 224])  // 根据模型输入尺寸调整
            .toFloat()
            .div(255.0)
            .expandDims();
    });
}

// 5. 文件上传处理
uploadBtn.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
        previewImg.src = e.target.result;
        
        await new Promise(resolve => previewImg.onload = resolve);
        
        showLoading("识别中...");
        try {
            const tensor = preprocessImage(previewImg);
            const predictions = await model.predict(tensor).data();
            displayResults(predictions);
        } catch (err) {
            showError(`识别失败: ${err.message}`);
        } finally {
            hideLoading();
        }
    };
    reader.readAsDataURL(file);
});

// 6. 显示结果（根据你的分类修改）
function displayResults(predictions) {
    const classNames = ["可回收", "有害", "厨余", "其他"]; // 替换为你的分类标签
    const maxProb = Math.max(...predictions);
    const predictedClass = classNames[predictions.indexOf(maxProb)];
    
    resultDiv.innerHTML = `
        <h3>识别结果：${predictedClass}</h3>
        <p>置信度：${(maxProb * 100).toFixed(1)}%</p>
    `;
}

// 辅助函数
function showLoading(msg) { /* ... */ }
function hideLoading() { /* ... */ }
function showError(msg) { /* ... */ }

// 初始化
loadModel();
