import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js';

// 1. 模型路径配置
const MODEL_PATH = 'https://raw.githack.com/zhongchenhui/garbage_classification/main/models/my_model/model.json'; // 路径相对于你的网页根目录
let model;

// 2. 加载模型函数
async function loadModel() {
    try {
        console.log('正在加载模型...');
        model = await tf.loadLayersModel(MODEL_PATH);
        console.log('模型加载成功');
    } catch (err) {
        console.error('模型加载失败:', err);
    }
}

// 3. 图片预处理函数
function preprocessImage(imgElement) {
    return tf.tidy(() => {
        // 调整尺寸需与模型训练时一致（这里假设是224x224）
        return tf.browser.fromPixels(imgElement)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(255.0)
            .expandDims();
    });
}

// 4. 预测函数
export async function predict(imageElement) {
    if (!model) {
        console.error('模型未加载！');
        return;
    }
    
    const tensor = preprocessImage(imageElement);
    const predictions = await model.predict(tensor).data();
    return predictions;
}

// 5. 初始化加载模型
loadModel();
