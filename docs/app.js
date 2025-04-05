// 初始化元素引用
const uploadInput = document.getElementById('upload');
const previewImg = document.getElementById('preview');
const resultDiv = document.getElementById('result');
const classLabels = ['可回收', '有害', '厨余', '其他'];
let model;

// 显示加载状态
function showLoading(text) {
  resultDiv.innerHTML = `<div class="loading">⏳ ${text}</div>`;
}

// 隐藏加载状态
function hideLoading() {
  const loading = document.querySelector('.loading');
  if (loading) loading.remove();
}

// 显示错误
function showError(text) {
  resultDiv.innerHTML = `<div class="error">X ${text}</div>`;
}

// 加载TensorFlow.js模型
async function loadModel() {
  try {
    showLoading('正在加载AI模型...');
    model = await tf.loadLayersModel('models/garbage_model/model.json');
    console.log('模型加载成功');
  } catch (err) {
    showError(`模型加载失败: ${err.message}`);
  } finally {
    hideLoading();
  }
}

// 图像预处理
function preprocessImage(img) {
  return tf.tidy(() => {
    return tf.browser.fromPixels(img)
      .resizeNearestNeighbor([224, 224])  // 必须与模型输入尺寸一致
      .toFloat()
      .div(255.0)
      .expandDims();
  });
}

// 处理文件上传
uploadInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async (e) => {
    previewImg.src = e.target.result;
    previewImg.style.display = 'block';

    // 等待图片加载完成
    await new Promise(resolve => previewImg.onload = resolve);
    
    try {
      showLoading('正在分析...');
      const tensor = preprocessImage(previewImg);
      const prediction = await model.predict(tensor).data();
      showPrediction(prediction);
    } catch (err) {
      showError(`分析失败: ${err.message}`);
    } finally {
      hideLoading();
    }
  };
  reader.readAsDataURL(file);
});

// 显示预测结果
function showPrediction(predictions) {
  const maxProb = Math.max(...predictions);
  const predictedIndex = predictions.indexOf(maxProb);
  
  resultDiv.innerHTML = `
    <h3>识别结果：${classLabels[predictedIndex]}</h3>
    <p>置信度：${(maxProb * 100).toFixed(1)}%</p>
    <div class="tips">${getDisposalTip(classLabels[predictedIndex])}</div>
  `;
}

// 获取处理建议
function getDisposalTip(className) {
  const tips = {
    '可回收': '请清洁后投入蓝色垃圾桶',
    '有害': '请投入红色专用回收箱',
    '厨余': '请去除塑料袋后投入绿色垃圾桶',
    '其他': '请投入灰色垃圾桶'
  };
  return tips[className] || '请按当地分类标准处理';
}

// 初始化
loadModel();
