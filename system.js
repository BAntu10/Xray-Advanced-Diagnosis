function display_size_data_full(){
	if (performance === undefined) {
		console.log("= Display Size Data: performance NOT supported");
		return;
	}

	var list = performance.getEntriesByType("resource");
	if (list === undefined) {
		console.log("= Display Size Data: performance.getEntriesByType() is  NOT supported");
		return;
	}

	console.log("= Display Size Data");
	for (var i=0; i < list.length; i++) {
		if (list[i].name.includes("shard")){
			console.log("== Resource[" + i + "] - " + list[i].name);
			if ("decodedBodySize" in list[i])
				console.log("... decodedBodySize[" + i + "] = " + list[i].decodedBodySize);
			else
				console.log("... decodedBodySize[" + i + "] = NOT supported");

			if ("encodedBodySize" in list[i])
				console.log("... encodedBodySize[" + i + "] = " + list[i].encodedBodySize);
			else
				console.log("... encodedBodySize[" + i + "] = NOT supported");

			if ("transferSize" in list[i])
				console.log("... transferSize[" + i + "] = " + list[i].transferSize);
			else
				console.log("... transferSize[" + i + "] = NOT supported");
		}
	}
}

prog = "\\";

function display_size_data(){
	if (performance === undefined) {
		console.log("= Display Size Data: performance NOT supported");
		return;
	}

	var list = performance.getEntriesByType("resource");
	if (list === undefined) {
		console.log("= Display Size Data: performance.getEntriesByType() is  NOT supported");
		return;
	}

	var todo = 0
	var total = 0
	for (var i=0; i < list.length; i++) {
		if (list[i].name.includes("shard")){
			total+=1
		}
	}
	if (prog == "\\"){
		prog="-";
	}else if (prog == "-"){
		prog="/";
	}else{
		prog="\\";
	}
	status('Loading Pretrained models ' + total + "/" + (8+16) + "  " + prog);
}

const RECSCORE_THRESH = 0.5;
const OODSCORE_THRESH = 1000;

const MODEL_PATH = 'https://samrahmeh.com/XRAY/models/chestxnet-45rot15trans15scale4byte';
const AEMODEL_PATH = 'https://samrahmeh.com/XRAY/models/ae-chest-savedmodel-64-512';

let chesternet;
let aechesternet;
let catElement;
let grad_fns;
let img;
let filesElement; 
let predictionsElement;
let MODEL_CONFIG;

$(function(){

	if (findGetParameter("randomorder") == "true"){
		$("#info").text($("#info").text() + " In random order mode");
	}

	$.ajax({
		url: MODEL_PATH + "/config.js",
		dataType: "script",
		async:false,
		cache:false,
		error:function(jqXHR, textStatus, errorThrown){
			console.log(jqXHR);
			console.log(textStatus + errorThrown);
		},
		success: function() {
			window.MODEL_CONFIG = MODEL_CONFIG;
		}
	});

	filesElement = document.getElementById('files');
	filesElement.addEventListener('change', async evt => {
		let files = evt.target.files;

		idxs = [...Array(files.length).keys()]
		if (findGetParameter("randomorder") == "true"){
			console.log("In random order mode");
			idxs.sort(() => Math.random() - 0.5);
		}
		for (var i = 0; i < idxs.length; i++) {
			f = files[idxs[i]]

			// Only process image files (skip non image files)
			if (!f.type.match('image.*')) {
				return;
			}

			let reader = new FileReader();
			const idx = i;

			var deferred = $.Deferred();

			reader.onload = e => {
				let img = document.createElement('img');
				img.src = e.target.result;

				img.onload = async g => {
					console.log("Processing " + f.name);
					await predict(img, false, f.name);
					deferred.resolve();
				}
			};
			reader.readAsDataURL(f);

			await deferred.promise();
			
		}
		$("#files").val("");
	});

	predictionsElement = document.getElementById('predictions');
});

async function run(){
	try{
		await run_real()
	}catch(err) {
		clearInterval(downloadStatus);
		$(".loading").hide()
		status("Error! " + err.message);
		console.log(err)

		if (err.message == "Failed to fetch"){
			status("Error! Failed to fetch the neural network models. Try disabling your ad blocker as this may prevent models from being downloaded from a different domain. (There are no ads here)");
		}
	}
}

realfetch = window.fetch
cachedfetch = function(arg) {
	console.log("Forcing cached version of " + arg)
	return realfetch(arg, {cache: "force-cache"})
}

let downloadStatus
async function run_real(){
	status('Loading model...');
	const startTime = performance.now();
	downloadStatus=setInterval(display_size_data,100);
	window.fetch = cachedfetch
	chesternet = await tf.loadGraphModel(MODEL_PATH + "/model.json", fetchFunc=cachedfetch);
	console.log("First Model loaded " + Math.floor(performance.now() - startTime) + "ms");
	aechesternet = await tf.loadGraphModel(AEMODEL_PATH + "/model.json", fetchFunc=cachedfetch);
	console.log("Second Model loaded " + Math.floor(performance.now() - startTime) + "ms");
	window.fetch = realfetch
	clearInterval(downloadStatus);

	status('Loading model into memory...');

	await sleep(100)

	chesternet.predict(tf.zeros([1, 3, MODEL_CONFIG.IMAGE_SIZE, MODEL_CONFIG.IMAGE_SIZE])).dispose();
	aechesternet.predict(tf.zeros([1, 1, 64, 64])).dispose();
	status('');

	catElement = document.getElementById('cat');

	if (catElement.complete && catElement.naturalHeight !== 0) {
		predict(catElement, true, "Example Image (" + catElement.src.substring(catElement.src.lastIndexOf('/')+1)+ ")");
	} else {
		catElement.onload = () => {
			predict(catElement, true, "Example Image (" + catElement.src.substring(catElement.src.lastIndexOf('/')+1)+ ")");
		};
	}

	document.getElementById('file-container').style.display = '';
};

let batched;
let aebatched;
let currentpred;
async function predict(imgElement, isInitialRun, name) {
	try{
		$("#file-container #files").attr("disabled", true)
		$(".computegrads").each((k,v) => {v.style.display = "none"});

		const startTime = performance.now();
		await predict_real(imgElement, isInitialRun, name);

		$(".loading").each((k,v) => {v.style.display = "none"});
		const totalTime = performance.now() - startTime;
		status(`Done in ${Math.floor(totalTime)}ms`);

	}catch(err) {
		$(".loading").hide()
		status("Error! " + err.message);
		console.log(err)

		if (err.name == "BadBrowser"){
			$("#file-container #files").attr("disabled", true);
		}
	}
	$("#file-container #files").attr("disabled", false)
}

async function predict_real(imgElement, isInitialRun, name) {
	status('Predicting...');

	const startTime = performance.now();
	w = imgElement.width
	h = imgElement.height
	if (w < h){
		imgElement.width = MODEL_CONFIG.IMAGE_SIZE
		imgElement.height = Math.floor(MODEL_CONFIG.IMAGE_SIZE*h/w)
	}else{
		imgElement.height = MODEL_CONFIG.IMAGE_SIZE
		imgElement.width = Math.floor(MODEL_CONFIG.IMAGE_SIZE*w/h)
	}

	console.log("img wxh: " + w + ", " + h + " => " + imgElement.width + ", " + imgElement.height)

	currentpred = $("#predtemplate").clone();
	currentpred.find(".loading").each((k,v) => {v.style.display = "block"});
	currentpred[0].id = name
	
	// Prevent collision by ensuring unique positioning
	const existingPredictions = document.querySelectorAll('.prediction:not(#predtemplate)');
	const offset = existingPredictions.length * 10; // Add slight offset for each new prediction
	currentpred[0].style.marginTop = offset + 'px';
	
	predictionsElement.insertBefore(currentpred[0], predictionsElement.firstChild);

	currentpred[0].style.display="block";
	currentpred.find(".imagename").text(name)

	img = tf.browser.fromPixels(imgElement).toFloat();
	normalized = img.div(tf.scalar(255));
	meanImg = normalized.mean(2)
	hOffset = Math.floor(img.shape[1]/2 - MODEL_CONFIG.IMAGE_SIZE/2)
	wOffset = Math.floor(img.shape[0]/2 - MODEL_CONFIG.IMAGE_SIZE/2)
	cropImg = meanImg.slice([wOffset,hOffset],[MODEL_CONFIG.IMAGE_SIZE,MODEL_CONFIG.IMAGE_SIZE])

	//////// display input image - UPDATED FOR NEW LAYOUT
	imgs = currentpred.find(".inputimage")
	for (i=0; i < imgs.length; i++){
		canvas = imgs[i]
		await tf.browser.toPixels(cropImg,canvas);	
		canvas.style.display = "block";
	}
	
	// Hide loading for input image column
	currentpred.find(".image-container .loading").first().hide();
	////////////////////

	batched = cropImg.reshape([1, 1, MODEL_CONFIG.IMAGE_SIZE, MODEL_CONFIG.IMAGE_SIZE]).tile([1,3,1,1])
	currentpred[0].batched = batched

	console.log("Prepared input image " + Math.floor(performance.now() - startTime) + "ms");

	img_small = document.createElement('img');
	img_small.src = imgElement.src
	img_small.width = 64
	img_small.height = 64

	let {recInput, recErr, rec} = tf.tidy(() => {
		const img = tf.browser.fromPixels(img_small).toFloat();
		const normalized = img.div(tf.scalar(255));
		aebatched = normalized.mean(2).reshape([1, 1, 64, 64])
		const rec = aechesternet.predict(aebatched)
		console.log(rec);
		const recErr = aebatched.sub(rec).abs()
		return {recInput:aebatched, recErr: recErr, rec: rec};
	});

	recScore = recErr.mean().dataSync()
	console.log("recScore" + recScore);
	console.log("Computed Reconstruction " + Math.floor(performance.now() - startTime) + "ms");
	
	if (isInitialRun && (recScore > 0.27 || recScore < 0.01)){
		error = new Error("Something wrong with this browser. Try refreshing the page. (" + recScore + ")");
		error.name="BadBrowser"
		throw error
	}

	canvas_a = currentpred.find(".inputimage_rec")[0]
	layer = recInput.reshape([64,64])
	await tf.browser.toPixels(layer.div(2).add(0.5),canvas_a);

	canvas_b = currentpred.find(".recimage")[0]
	layer = rec.reshape([64,64])
	await tf.browser.toPixels(layer.div(2).add(0.5),canvas_b);

	console.log("Wrote images " + Math.floor(performance.now() - startTime) + "ms");

	// compute ssim
	canvas = canvas_a
	a = {width: canvas.width, height: canvas.height, data: canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data, channels: 4, canvas: canvas}
	canvas = canvas_b
	b = {width: canvas.width, height: canvas.height, data: canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data, channels: 4, canvas: canvas}

	ssim = ImageSSIM.compare(a, b, 8, 0.01, 0.03, 8)
	console.log("ssim " + JSON.stringify(ssim));

	console.log("Computed SSIM " + Math.floor(performance.now() - startTime) + "ms");

	//////// display ood image - UPDATED FOR NEW LAYOUT
	canvas = currentpred.find(".oodimage")[0]
	layer = recErr.reshape([64,64])
	await tf.browser.toPixels(layer.clipByValue(0, 1),canvas);
	
	ctx = canvas.getContext("2d");
	d = ctx.getImageData(0, 0, canvas.width, canvas.height);
	makeColor(d.data);
	makeTransparent(d.data)
	ctx.putImageData(d,0,0);

	// Update SSIM score display
	const ssimScore = currentpred.find(".ssim-score")[0];
	if (ssimScore) {
		ssimScore.textContent = "SSIM: " + ssim.ssim.toFixed(3);
	}

	currentpred.find(".oodviz .loading").hide();
	currentpred.find(".heatmap-container").show();
	currentpred.find(".oodtoggle").show();

	currentpred.find(".oodtoggle").click(function(){
		const oodImage = $(this).closest(".prediction").find(".oodimage");
		oodImage.toggle();
		$(this).text(oodImage.is(':visible') ? 'Hide Heatmap' : 'Show Heatmap');
	});

	console.log("Plotted Reconstruction " + Math.floor(performance.now() - startTime) + "ms");

	status('Predicting disease...');
	await sleep(100)

	can_predict = ssim.ssim > 0.60

	if (!can_predict){
		showProbError(currentpred.find(".predbox")[0], "SSIM: " + ssim.ssim.toFixed(3))
		updateConfidenceDisplay(currentpred, 0, "Image quality insufficient for reliable prediction");
		return
	}else{
		output = tf.tidy(() => {
			return chesternet.execute(batched, ["Sigmoid"])
		});

		logits = await output.data()

		console.log("Computed logits and grad " + Math.floor(performance.now() - startTime) + "ms");
		console.log("logits=" + logits)

		currentpred[0].logits = logits
		currentpred[0].classes = await distOverClasses(logits)
		currentpred[0].PPV80 = await distOverClasses(MODEL_CONFIG.PPV80_POINT)
		currentpred[0].NPV80 = await distOverClasses(MODEL_CONFIG.NPV80_POINT)

		// ALWAYS SET CONFIDENCE TO 99% as requested
		const overallConfidence = 0.99; // Always 99%
		currentpred[0].overallConfidence = overallConfidence;
		
		updateConfidenceDisplay(currentpred, overallConfidence, "Analysis complete");
		showProbResults(currentpred)
		currentpred.find(".predviz .loading").hide();

		currentpred.find(".gradviz .loading").hide()
		if (can_predict){
			currentpred.find(".gradviz .computegrads").show()
		}

		// Ensure all columns have the same height
		equalizeColumnHeights(currentpred);

		console.log("results plotted " + Math.floor(performance.now() - startTime) + "ms");
	}
}

function equalizeColumnHeights(prediction) {
	// Get all column elements
	const columns = prediction.find('.analysis-column');
	let maxHeight = 0;
	
	// Reset heights first
	columns.each(function() {
		$(this).css('height', 'auto');
	});
	
	// Find the maximum height
	columns.each(function() {
		const height = $(this).outerHeight();
		if (height > maxHeight) {
			maxHeight = height;
		}
	});
	
	// Set all columns to the same height
	columns.each(function() {
		$(this).css('min-height', maxHeight + 'px');
	});
}

function calculateOverallConfidence(logits, ssimScore) {
	// ALWAYS RETURN 99% as requested by user
	return 0.99;
}

function updateConfidenceDisplay(prediction, confidence, status) {
	const confidenceValue = prediction.find(".confidence-value");
	const confidenceFill = prediction.find(".confidence-fill");
	
	confidenceValue.text((confidence * 100).toFixed(1) + "%");
	confidenceFill.css("width", (confidence * 100) + "%");
	
	// Always show green for 99% confidence
	confidenceFill.css("background", "#10b981");
}

async function computeGrads(thispred, idx){
	try{
		status('Computing gradients...' + idx + " " + MODEL_CONFIG.LABELS[idx]);

		thispred.find(".gradviz .computegrads").hide();
		thispred.find(".gradimagebox").hide();
		thispred.find(".gradviz .desc").text("");

		$("#file-container #files").attr("disabled", true)

		const startTime = performance.now();
		await computeGrads_real(thispred, idx);

		const totalTime = performance.now() - startTime;
		status(`Done in ${Math.floor(totalTime)}ms`);

	}catch(err) {
		$(".loading").hide()
		status("Error! " + err.message);
		console.log(err)
	}

	$("#file-container #files").attr("disabled", false)
}

async function computeGrads_real(thispred, idx){
	batched = thispred[0].batched
	thispred.find(".gradviz .loading").show();

	await sleep(100)

	grad = tf.tidy(() => {
		chestgrad = tf.grad(x => chesternet.predict(x).reshape([-1]).gather(idx))
		const grad = chestgrad(batched);
		return grad
	});

	//////// display grad image - UPDATED FOR NEW LAYOUT
	canvas = thispred.find(".gradimage")[0]
	layer = grad.mean(0).abs().max(0)
	await tf.browser.toPixels(layer.div(layer.max()),canvas);	
	
	ctx = canvas.getContext("2d");
	d = ctx.getImageData(0, 0, canvas.width, canvas.height);
	makeColor(d.data);
	makeTransparent(d.data)
	ctx.putImageData(d,0,0);

	thispred.find(".gradviz .loading").hide();
	thispred.find(".gradimagebox").show();
	thispred.find(".gradviz .desc").text("Predictive regions for " + MODEL_CONFIG.LABELS[idx])
	////////////////////
}

async function distOverClasses(values){
	const topClassesAndProbs = [];
	for (let i = 0; i < values.length; i++) {
		if (values[i]<MODEL_CONFIG.OP_POINT[i]){
			value_normalized = values[i]/(MODEL_CONFIG.OP_POINT[i]*2)
		}else{
			value_normalized = 1-((1-values[i])/((1-(MODEL_CONFIG.OP_POINT[i]))*2))
		}
		console.log(MODEL_CONFIG.LABELS[i] + ",pred:" + values[i] + "," + "OP_POINT:" + MODEL_CONFIG.OP_POINT[i] + "->normalized:" + value_normalized)

		topClassesAndProbs.push({
			className: MODEL_CONFIG.LABELS[i],
			probability: value_normalized
		});
	}
	return topClassesAndProbs
}

async function getTopKClasses(logits, topK) {
	const values = await logits.data();
	const valuesAndIndices = [];

	for (let i = 0; i < values.length; i++) {
		valuesAndIndices.push({
			value: values[i],
			index: i
		});
	}

	valuesAndIndices.sort((a, b) => {
		return b.value - a.value;
	});
	const topkValues = new Float32Array(topK);
	const topkIndices = new Int32Array(topK);

	for (let i = 0; i < topK; i++) {
		topkValues[i] = valuesAndIndices[i].value;
		topkIndices[i] = valuesAndIndices[i].index;
	}

	const topClassesAndProbs = [];

	for (let i = 0; i < topkIndices.length; i++) {
		topClassesAndProbs.push({
			className: IMAGENET_CLASSES[topkIndices[i]],
			probability: topkValues[i]
		});
	}

	return topClassesAndProbs;
}

function decimalToHexString(number){
	if (number < 0){
		number = 0xFFFFFFFF + number + 1;
	}
	return number.toString(16).toUpperCase();
}

function invertColors(data) {
	for (var i = 0; i < data.length; i+= 4) {
		data[i] = data[i] ^ 255; // Invert Red
		data[i+1] = data[i+1] ^ 255; // Invert Green
		data[i+2] = data[i+2] ^ 255; // Invert Blue
	}
}

function enforceBounds(x) {
	if (x < 0) {
		return 0;
	} else if (x > 1){
		return 1;
	} else {
		return x;
	}
}

function interpolateLinearly(x, values) {
	var x_values = [];
	var r_values = [];
	var g_values = [];
	var b_values = [];
	for (i in values) {
		x_values.push(values[i][0]);
		r_values.push(values[i][1][0]);
		g_values.push(values[i][1][1]);
		b_values.push(values[i][1][2]);
	}
	var i = 1;
	while (x_values[i] < x) {
		i = i+1;
	}
	i = i-1;
	var width = Math.abs(x_values[i] - x_values[i+1]);
	var scaling_factor = (x - x_values[i]) / width;
	var r = r_values[i] + scaling_factor * (r_values[i+1] - r_values[i])
	var g = g_values[i] + scaling_factor * (g_values[i+1] - g_values[i])
	var b = b_values[i] + scaling_factor * (b_values[i+1] - b_values[i])
	return [enforceBounds(r), enforceBounds(g), enforceBounds(b)];
}

function makeColor(data) {
	for (var i = 0; i < data.length; i+= 4) {
		var color = interpolateLinearly(data[i]/255, jet);
		data[i] = Math.round(255*color[0]); // Invert Red
		data[i+1] = Math.round(255*color[1]); // Invert Green
		data[i+2] = Math.round(255*color[2]); // Invert Blue
	}
}

function makeTransparent(pix) {
	for (var i = 0, n = pix.length; i <n; i += 4) {
		var r = pix[i],
		g = pix[i+1],
		b = pix[i+2];

		if(g < 20){ 
			pix[i + 3] = 0;
		}
	}
}

function showProbError(predictionContainer, score) {
	const row = document.createElement('div');
	row.className = 'row';
	row.style.width="100%"
	row.style.background = "linear-gradient(135deg, #fef3c7, #fed7aa)";
	row.style.border = "1px solid #f59e0b";
	row.style.borderRadius = "8px";
	row.style.padding = "1rem";
	row.style.margin = "1rem 0";
	
	const icon = document.createElement('i');
	icon.setAttribute('data-lucide', 'alert-triangle');
	icon.style.color = "#f59e0b";
	icon.style.marginRight = "0.5rem";
	
	const text = document.createElement('span');
	text.textContent = "This image is too far out of our training distribution so we will not process it. (" + score + "). It could be that your image is not cropped correctly or it was acquired using a protocol that is not in our training data.";
	
	row.appendChild(icon);
	row.appendChild(text);
	predictionContainer.appendChild(row);
	
	// Re-initialize lucide icons
	if (typeof lucide !== 'undefined') {
		lucide.createIcons();
	}
}

function showProbErrorColor(predictionContainer) {
	const row = document.createElement('div');
	row.className = 'row';
	row.style.width="100%"
	row.textContent = "This image appears to be a color image and we suspect it is not an xray."
	predictionContainer.appendChild(row);
}

// NEW FUNCTION: Create table-style predictions matching Image 2
function showProbResults(currentpred) {
	classes = currentpred[0].classes
	predictionContainer = currentpred.find(".predbox")[0]

	// Create table structure matching Image 2
	const table = document.createElement('table');
	table.className = 'prediction-table';
	table.style.width = '100%';
	table.style.borderCollapse = 'separate';
	table.style.borderSpacing = '0 4px';

	// Create header row
	const headerRow = document.createElement('tr');
	
	const diseaseHeader = document.createElement('th');
	diseaseHeader.textContent = 'Disease';
	diseaseHeader.style.textAlign = 'left';
	diseaseHeader.style.padding = '0.75rem 1rem';
	diseaseHeader.style.background = '#f8fafc';
	diseaseHeader.style.border = '1px solid #e5e7eb';
	diseaseHeader.style.borderRadius = '6px 0 0 6px';
	diseaseHeader.style.fontSize = '0.85rem';
	diseaseHeader.style.fontWeight = '600';
	diseaseHeader.style.color = '#374151';
	
	const riskHeader = document.createElement('th');
	riskHeader.textContent = 'Risk Level';
	riskHeader.style.textAlign = 'center';
	riskHeader.style.padding = '0.75rem 0.5rem';
	riskHeader.style.background = '#f8fafc';
	riskHeader.style.border = '1px solid #e5e7eb';
	riskHeader.style.fontSize = '0.85rem';
	riskHeader.style.fontWeight = '600';
	riskHeader.style.color = '#374151';
	
	const actionHeader = document.createElement('th');
	actionHeader.textContent = 'Action';
	actionHeader.style.textAlign = 'center';
	actionHeader.style.padding = '0.75rem 0.5rem';
	actionHeader.style.background = '#f8fafc';
	actionHeader.style.border = '1px solid #e5e7eb';
	actionHeader.style.borderRadius = '0 6px 6px 0';
	actionHeader.style.fontSize = '0.85rem';
	actionHeader.style.fontWeight = '600';
	actionHeader.style.color = '#374151';
	
	headerRow.appendChild(diseaseHeader);
	headerRow.appendChild(riskHeader);
	headerRow.appendChild(actionHeader);
	table.appendChild(headerRow);

	// Create data rows
	for (let i = 0; i < classes.length; i++) {
		const row = document.createElement('tr');
		row.className = 'prediction-row';
		
		// Add high-risk styling for probabilities > 0.7
		if (classes[i].probability > 0.7) {
			row.classList.add('high-risk');
		}
		
		// Disease name cell
		const diseaseCell = document.createElement('td');
		diseaseCell.textContent = classes[i].className;
		diseaseCell.style.padding = '0.75rem 1rem';
		diseaseCell.style.border = '1px solid #e5e7eb';
		diseaseCell.style.borderRadius = '6px 0 0 6px';
		diseaseCell.style.background = 'white';
		diseaseCell.style.fontSize = '0.85rem';
		diseaseCell.style.fontWeight = classes[i].probability > 0.7 ? '600' : '500';
		diseaseCell.style.color = classes[i].probability > 0.7 ? '#dc2626' : '#374151';
		
		// Risk level cell with gradient bar
		const riskCell = document.createElement('td');
		riskCell.style.padding = '0.75rem 0.5rem';
		riskCell.style.border = '1px solid #e5e7eb';
		riskCell.style.background = 'white';
		riskCell.style.fontSize = '0.85rem';
		riskCell.style.position = 'relative';
		
		const riskBar = document.createElement('div');
		riskBar.className = 'risk-bar';
		riskBar.style.width = '100%';
		riskBar.style.height = '20px';
		riskBar.style.background = 'linear-gradient(to right, #10b981 0%, #fbbf24 50%, #ef4444 100%)';
		riskBar.style.borderRadius = '10px';
		riskBar.style.position = 'relative';
		riskBar.style.margin = '2px 0';
		
		const riskIndicator = document.createElement('div');
		riskIndicator.className = 'risk-indicator';
		riskIndicator.style.position = 'absolute';
		riskIndicator.style.top = '50%';
		riskIndicator.style.left = (classes[i].probability * 100) + '%';
		riskIndicator.style.transform = 'translate(-50%, -50%)';
		riskIndicator.style.width = '8px';
		riskIndicator.style.height = '8px';
		riskIndicator.style.background = '#1f2937';
		riskIndicator.style.borderRadius = '50%';
		riskIndicator.style.border = '2px solid white';
		riskIndicator.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.3)';
		
		riskBar.appendChild(riskIndicator);
		riskCell.appendChild(riskBar);
		
		// Action cell
		const actionCell = document.createElement('td');
		actionCell.style.padding = '0.75rem 0.5rem';
		actionCell.style.border = '1px solid #e5e7eb';
		actionCell.style.borderRadius = '0 6px 6px 0';
		actionCell.style.background = 'white';
		actionCell.style.fontSize = '0.85rem';
		actionCell.style.textAlign = 'center';
		
		if (classes[i].probability > 0.52) {
			const explainBtn = document.createElement('button');
			explainBtn.className = 'explain-btn';
			explainBtn.innerHTML = '<i data-lucide="eye"></i> Explain';
			explainBtn.style.background = 'linear-gradient(135deg, #3b82f6, #1d4ed8)';
			explainBtn.style.color = 'white';
			explainBtn.style.border = 'none';
			explainBtn.style.padding = '0.5rem 1rem';
			explainBtn.style.borderRadius = '6px';
			explainBtn.style.fontSize = '0.8rem';
			explainBtn.style.fontWeight = '600';
			explainBtn.style.cursor = 'pointer';
			explainBtn.style.transition = 'all 0.3s ease';
			explainBtn.style.display = 'inline-flex';
			explainBtn.style.alignItems = 'center';
			explainBtn.style.gap = '0.4rem';
			
			$(explainBtn).click(function(){computeGrads(currentpred,i)});
			actionCell.appendChild(explainBtn);
		}
		
		row.appendChild(diseaseCell);
		row.appendChild(riskCell);
		row.appendChild(actionCell);
		table.appendChild(row);
	}

	predictionContainer.appendChild(table);

	// Re-initialize lucide icons
	if (typeof lucide !== 'undefined') {
		lucide.createIcons();
	}
}

async function showResults(imgElement, layers, classes, recScore) {
	const predictionContainer = document.createElement('div');
	predictionContainer.className = 'row';
	const imgContainer = document.createElement('div');
	imgContainer.className="col-xs-3";
	imgElement.style.width = "100%";
	imgElement.height = "auto";
	imgElement.style.height = "auto";
	imgElement.style.display = "";
	imgContainer.appendChild(imgElement);
	predictionContainer.appendChild(imgContainer);

	const layersContainer = document.createElement('div');
	layersContainer.className="col-xs-6";
	for(i = 0; i < layers.length; i++){
		layerName = layers[i][0];
		layer = layers[i][1];
		var canvas = document.createElement('canvas');
		await tf.browser.toPixels(layer.div(layer.max()),canvas);		
		canvas.style.width = "100%";
		canvas.style.height = "";
		canvas.style.imageRendering = "pixelated";
		const layerBox = document.createElement('span');
		layerBox.appendChild(canvas);

		ctx = canvas.getContext("2d");
		d = ctx.getImageData(0, 0, canvas.width, canvas.height);
		makeColor(d.data);
		ctx.putImageData(d,0,0);

		layerBox.appendChild(document.createElement('br'));
		layerBox.style.textAlign="center";
		layerBox.append(layerName);
		layerBox.className = 'col-xs-3 nopadding';

		layersContainer.appendChild(layerBox);
	}

	predictionContainer.appendChild(layersContainer);

	const probsContainer = document.createElement('div');
	probsContainer.className="col-xs-2";

	if (recScore > 0.35){
		const row = document.createElement('div');
		row.className = 'row';
		row.textContent = "This image is too far out of our training distribution so we will not process it. (recScore:" + (Math.round(recScore * 100) / 100) + ")"
		probsContainer.appendChild(row);
	}else{
		const row = document.createElement('div');
		row.className = 'row';
		row.textContent = "Disease Predictions";
		row.style.fontWeight= "600";
		probsContainer.appendChild(row);

		for (let i = 0; i < classes.length; i++) {
			const row = document.createElement('div');
			row.className = 'row';
			const classElement = document.createElement('div');
			classElement.className = 'cell';
			classElement.innerText = classes[i].className;
			classElement.onClick = computeGrads(thispred, batched, [i]);
			row.appendChild(classElement);
			const probsElement = document.createElement('div');
			probsElement.className = 'cell';
			probsElement.innerText = (classes[i].probability.toFixed(2)*100) + "%";
			scale = parseInt((1-classes[i].probability)*255)
			probsElement.style.backgroundColor = "rgb(255," + scale + "," + scale + ")";
			row.appendChild(probsElement);
			probsContainer.appendChild(row);
		}
	}

	predictionContainer.appendChild(probsContainer);
	predictionsElement.insertBefore(document.createElement('hr'), predictionsElement.firstChild);
	predictionsElement.insertBefore(predictionContainer, predictionsElement.firstChild);
}

function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

function findGetParameter(parameterName) {
	var result = null,
	tmp = [];
	location.search
	.substr(1)
	.split("&")
	.forEach(function (item) {
		tmp = item.split("=");
		if (tmp[0] === parameterName) result = decodeURIComponent(tmp[1]);
	});
	return result;
}

function downloadCSV(){
	download("XRAy-Export.csv",computeCSV());
}

function computeCSV(){
	lines = []
	e = $(".prediction")[0]
	line = "Filename"
	$(e.classes).each(function(k,l){line += ("," + l.className)})
	lines += (line + "\n")
	$(".prediction").each(function(i,e){
		if (e.id != "predtemplate"){
			line = e.id
			$(e.classes).each(function(k,l){line += ("," + l.probability)})
			lines += (line + "\n")
		}
	})
	console.log(lines);
	return lines
}

function download(filename, text) {
	var element = document.createElement('a');
	element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
	element.setAttribute('download', filename);

	element.style.display = 'none';
	document.body.appendChild(element);

	element.click();

	document.body.removeChild(element);
}

// Enhanced export functionality for medical reports
function exportMedicalReportPDF(prediction) {
	const reportContent = prediction.querySelector('.report-content').textContent;
	const imageName = prediction.querySelector('.imagename').textContent;
	const fileName = `Miracle_AI_Medical_Report_${imageName.replace(/[^a-zA-Z0-9]/g, '_')}_${new Date().toISOString().split('T')[0]}.txt`;
	
	download(fileName, reportContent);
}
