import sys
sys.path.append('./ESRGAN')
sys.path.append('/StableSR')
import openai
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import subprocess
import random
import os
import requests
import numpy as np
import cv2
from openai import OpenAI
from PIL import Image
from einops import rearrange
from MPRNet.deblurring import process_image as deblurring
from MPRNet.denoising import process_image as denoising
from MPRNet.deraining import process_image as deraining
from PIL import Image
import torchvision
from ESRGAN.inference_realesrgan import process_image as esrgan
from ESRGAN.inference_realesrgan import initialize_model as esrgan_model_init
from StableSR.scripts.sr_val_ddpm_text_T_vqganfin_old import process_image as diffusionSR
from StableSR.scripts.sr_val_ddpm_text_T_vqganfin_old import initialize_models as diffusion_model_init
from StableSR.scripts.sr_val_ddpm_text_T_vqganfin_old import load_img as diffusionSR_load_img
from StableSR.scripts.sr_val_ddpm_text_T_vqganfin_old import parse
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import time
import cloudinary
import cloudinary.uploader
import cloudinary.api
import logging



client = OpenAI(api_key='Your_api_key')
# Configure Cloudinary with your credentials
cloudinary.config(
  cloud_name = 'dufmn1aaa',  # Replace with your cloud name
  api_key = 'xxxxxxxxxxxxxxx',        # Replace with your API key
  api_secret = '4xaqMk9n6RC7auTdOqKoexxx_xx'   # Replace with your API secret
)

def check_image_size(image_path):
    size = os.path.getsize(image_path)
    return size <= 20 * 1024 * 1024  # 20 MB
def convert_to_supported_format(image_path, target_format='.png'):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    new_image_path = os.path.splitext(image_path)[0] + target_format
    cv2.imwrite(new_image_path, image)
    return new_image_path
def save_population_and_fitness(population, fitness_scores, generation):
    # Save population
    with open(f'results/population_gen_{generation}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for individual in population:
            writer.writerow(individual)

    # Save fitness scores
    with open(f'results/fitness_scores_gen_{generation}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fitness_scores)


def plot_results(num_generations):
    # Check for available serif fonts and set font properties
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
    if 'Times New Roman' in available_fonts:
        font_name = 'Times New Roman'
    else:
        font_name = 'DejaVu Serif'

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': font_name,
        'font.size': 10
    })

    # Plot fitness scores
    fitness_scores_all = []

    for generation in range(num_generations):
        with open(f'results/fitness_scores_gen_{generation}.csv', 'r') as f:
            reader = csv.reader(f)
            scores = next(reader)
            scores = [float(score) for score in scores]
            fitness_scores_all.append(max(scores))

    plt.figure()
    plt.plot(range(num_generations), fitness_scores_all, marker='o')
    plt.title('Best Fitness Score per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.grid(True)
    plt.savefig(f'fitness_scores.png', dpi=300)
    plt.show()

    # Mapping of gene strings to unique colors
    gene_mapping = {}
    colors = list(mcolors.TABLEAU_COLORS.values())  # Use color values instead of keys
    current_gene_id = 0

    # Collecting data for all generations
    all_generations_population = []

    for generation in range(num_generations):
        population = []
        with open(f'results/population_gen_{generation}.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                encoded_row = []
                for gene in row:
                    if gene not in gene_mapping:
                        gene_mapping[gene] = colors[current_gene_id % len(colors)]
                        current_gene_id += 1
                    encoded_row.append(gene_mapping[gene])
                population.append(encoded_row)
        all_generations_population.append(population)

    # Plotting all generations on the same plot
    fig, ax = plt.subplots()
    rect_width = 0.4
    rect_height = 0.1
    gene_gap = 0.2
    for generation in range(num_generations):
        population = all_generations_population[generation]
        for person_idx, individual in enumerate(population):
            for gene_idx, gene in enumerate(individual):
                rect = Rectangle((generation - rect_width / 2, person_idx + gene_idx * gene_gap - gene_gap / 2),
                                 rect_width, rect_height, color=gene)
                ax.add_patch(rect)

    # Adding the legend
    legend_elements = [Rectangle((0, 0), 1, 1, color=color, label=gene) for gene, color in gene_mapping.items()]
    plt.legend(handles=legend_elements, title='Genes')

    plt.title('Gene Distribution Across Generations')
    plt.xlabel('Generation')
    plt.ylabel('Individual')
    plt.grid(True)
    plt.xlim(-0.5, num_generations - 0.5)
    plt.ylim(-0.5, len(all_generations_population[0]) - 0.5)
    plt.savefig(f'gene_distribution.png', dpi=300)
    plt.show()





# Upload an image to Cloudinary
def upload_image_to_cloudinary(image_path, retries=3, wait_time=5):
    for attempt in range(retries):
        try:
            response = cloudinary.uploader.upload(image_path)
            logging.info(f"Attempt {attempt + 1}: Image uploaded successfully")
            return response['secure_url']
        except cloudinary.exceptions.Error as e:
            logging.error(f"Failed to upload image: {str(e)}")
            if "Invalid cloud_name" in str(e):
                raise Exception("Invalid Cloudinary credentials. Please check your cloud_name, api_key, and api_secret.")
            if attempt < retries - 1:
                logging.warning("Retrying...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to upload image after {retries} attempts.")
                raise Exception(f"Failed to upload image after {retries} attempts.")



def upload_image_to_imgur(image_path, client_id, retries=3, wait_time=5):
    url = "https://api.imgur.com/3/upload"
    headers = {"Authorization": f"Client-ID {client_id}"}
    with open(image_path, "rb") as image_file:
        payload = {"image": image_file.read()}

    for attempt in range(retries):
        response = requests.post(url, headers=headers, data=payload)
        logging.info(f"Attempt {attempt + 1}: Status Code: {response.status_code}")
        if response.status_code == 200:
            return response.json()["data"]["link"]
        elif response.status_code == 429:
            logging.warning("Rate limit exceeded. Retrying...")
            time.sleep(wait_time)
        else:
            logging.error(f"Failed to upload image: {response.status_code}, {response.text}")
            if response.status_code == 503:
                logging.warning("Server error. Retrying...")
                time.sleep(wait_time)
    raise Exception(f"Failed to upload image after {retries} attempts.")


def upload_images_from_directory(directory_path):
    image_urls = {}
    image_paths = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(directory_path, filename)
            try:
                # image_url = upload_image_to_imgur(image_path, client_id)
                image_url = upload_image_to_cloudinary(image_path)
                image_urls[filename] = image_url
                image_paths[filename] = image_path
                print(f"Image uploaded successfully: {image_url}")
            except Exception as e:
                print(f"Failed to upload {filename}: {str(e)}")
                image_urls = {'input.png': 'https://i.imgur.com/PI9dlT8.png'}
                image_paths = {'input.png': 'inputs/input_rz.png'}
    return image_urls, image_paths


def save_img(filepath, img):
    # Ensure the image is in the correct format (uint8)
    if img.dtype != np.uint8:
        # Normalize the image to 0-255 range
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to uint8
        img = img.astype(np.uint8)

    # Convert from RGB to BGR color space
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Save the image
    cv2.imwrite(filepath, img_bgr)

# Function to apply DNN model (placeholder)
def apply_deblurring(image_path):
    # Process the image
    processed_image_path = deblurring(image_path)
    return processed_image_path

def apply_denoising(image_path):
    # Apply denoising model to the image
    processed_image_path = denoising(image_path)
    return processed_image_path

def apply_deraining(image_path):
    # Apply deraining model to the image
    processed_image_path = deraining(image_path)
    return processed_image_path

def apply_sharpening(image_path):
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])

    # Apply the sharpening kernel to the image
    image = cv2.imread(image_path)
    processed_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    processed_image_path = 'results/processed_image_sharpen.png'
    cv2.imwrite(processed_image_path,processed_image)
    # save_img(processed_image_path,processed_image)
    return processed_image_path

def apply_esrgan(image_path, saved_path = 'results/processed_image_esrgan.png'):
    # Apply ESRGAN model to the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    model = esrgan_model_init(model_name='RealESRGAN_x4plus')
    processed_image = esrgan(model, image)
    processed_image_path = saved_path
    cv2.imwrite(processed_image_path, processed_image)
    # save_img(processed_image_path,processed_image)
    return processed_image_path

# Function to apply StableDiffusion model (placeholder)
def apply_stable_diffusion(image_path, saved_path = 'results/processed_image_diffusion.png', ddpm_steps=[500]):
    # Apply StableDiffusion model to the image
    image = diffusionSR_load_img(image_path)
    vqgan_config_path, vqgan_ckpt_path, model_config_path, model_ckpt_path, device, opt = parse()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.input_size),
        torchvision.transforms.CenterCrop(opt.input_size),
    ])
    image = transform(image)
    image = image.clamp(-1, 1)


    vq_model, model = diffusion_model_init(vqgan_config_path, vqgan_ckpt_path, model_config_path, model_ckpt_path, device, opt.dec_w)
    processed_image = diffusionSR(image, vq_model, model, ddpm_steps=ddpm_steps)
    processed_image_path = saved_path
    x_sample = 255. * rearrange(processed_image.cpu().detach().numpy().squeeze(0), 'c h w -> h w c')
    Image.fromarray(x_sample.astype(np.uint8)).save(processed_image_path)
    # save_img(processed_image_path, x_sample.astype(np.uint8))
    return processed_image_path

# Load your test image
# test_image_path = "inputs/input.png"
# original_image = Image.open(test_image_path)


# Function to get processing ratings from ChatGPT-4
def get_image_quality_score2(filename, url):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": f"Please rate the need for deblurring, denoising, sharpening, realism, and perceptual quality for the given image {filename}. \
                 Provide each rating on a scale from 1 to 10, and return the ratings as a one-dimensional array with six elements inside two $. \
                 For example: $5, 2, 5, 9, 8$."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": "auto"  # or "low" / "high" based on your need
                    }
                }
            ]
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=50,
        )
        response_openai = response.choices[0].message.content
        score = response_openai.split('$')[1]  # .split('$')[0]
        # Step 1: Split the string into a list of strings
        score_list = score.split(',')
        # Step 2: Convert the list of strings into a list of integers
        int_list = [int(num) for num in score_list]
        # Step 3: Convert the list of integers into a NumPy array
        score = np.array(int_list)
        print(response_openai)

        # engine.say(response_openai)
        # engine.runAndWait()
        return score
    except (IndexError, ValueError, KeyError) as e:
        print(f"Error parsing scores from response: {response_openai}")
        # print(f"Exception: {e}")
        # Return a zero score array if parsing fails
        return np.array([0, 0, 0, 0])
    # except (IndexError, ValueError, KeyError) as e:
    #     print(f"Error parsing scores from response: {response_openai}")
    #     # print(f"Exception: {e}")
    #     # Return a zero score array if parsing fails
    #     return np.array([0, 0, 0, 0])

def get_image_quality_score(filename, url):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": f"Do you think this image {filename} has the good quality? Give me a score from 0 to 10 with the sentence format of 'Image score = X out of 10 at first'."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": "auto"  # or "low" / "high" based on your need
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
    )
    response_openai = response.choices[0].message.content
    score = int(response_openai.split('Image score = ')[1].split(' out of 10 at first')[0])
    print(response_openai)
    # engine.say(response_openai)
    # engine.runAndWait()
    # Convert the response text to speech
    # tts = gTTS(text=response_openai, lang='en')
    # tts.save(remote_audio_file)# Transfer the file and create the HTML file
    # html_file_path = transfer_file_to_google_mac()
    # # Load the converted speech
    # pygame.mixer.music.load("./response.mp3")
    # # Play the converted speech
    # pygame.mixer.music.play()
    # # Keep the script running until the sound is finished
    # while pygame.mixer.music.get_busy():
    #     time.sleep(1)
    return score


# Genetic Algorithm Components
def initialize_population(pop_size, num_functions):
    population = []
    for _ in range(pop_size):
        individual = random.choices(['deblurring', 'denoising', 'deraining', 'sharpening', 'esrgan', 'diffusion', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty'], k=num_functions)
        population.append(individual)
    return population
def initialize_population(pop_size, num_functions):
    genes = ['deblurring', 'denoising', 'deraining', 'sharpening', 'esrgan', 'diffusion', 'Empty', 'Empty', 'Empty']
    population = []
    for _ in range(pop_size):
        individual = []
        while len(individual) < num_functions:
            gene = random.choice(genes)
            if len(individual) == 0 or gene != individual[-1]:
                individual.append(gene)
        population.append(individual)
    return population

# Function to initialize the population with specific combinations
def initialize_population_with_combinations(pop_size, num_functions, genes):
    genes = ['deblurring', 'denoising', 'deraining', 'sharpening']
    population = []
    for _ in range(pop_size):
        individual = []
        while len(individual) < num_functions:
            gene = random.choice(genes)
            if len(individual) == 0 or gene != individual[-1]:
                individual.append(gene)
        population.append(individual)
    return population

def initialize_population_from_specific_combinations(pop_size, specific_combinations):
    population = []
    while len(population) < pop_size:
        # Randomly select a combination from the specific combinations
        combination = random.choice(specific_combinations)
        population.append(combination)

    return population

specific_combinations = [
    ['denoising', 'deblurring', 'diffusion', 'esrgan'],
    ['sharpening', 'esrgan', 'sharpening', 'deraining'],
    ['denoising', 'sharpening', 'deblurring', 'denoising'],
    ['diffusion', 'deblurring', 'esrgan', 'deraining'],
    ['sharpening', 'esrgan', 'sharpening', 'deraining'],
    ['sharpening', 'esrgan', 'sharpening', 'deraining'],
    ['denoising', 'sharpening', 'deblurring', 'denoising'],
    ['denoising', 'sharpening', 'deblurring', 'denoising']
]


def save_processed_image(processed_image_path, subdirectory, filename):
    # Create the subdirectory if it doesn't exist
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    # Define the full path for the new image
    new_image_path = os.path.join(subdirectory, filename)

    # Load the image using cv2
    image = cv2.imread(processed_image_path)

    # Save the image to the new path
    cv2.imwrite(new_image_path, image)

    #return new_image_path

def is_image_file(filename):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    return filename.lower().endswith(valid_extensions)


def get_random_images(image_dir, num_images=2):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                   os.path.isfile(os.path.join(image_dir, f)) and is_image_file(f)]

    if len(image_paths) < num_images:
        raise ValueError("Not enough images in the directory to select the requested number.")

    random_images = random.sample(image_paths, num_images)
    return random_images
def evaluate_fitness(individual, image_dir, generation):
    # List all images in the directory
    selected_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                   os.path.isfile(os.path.join(image_dir, f)) and is_image_file(f)]
    # selected_images = get_random_images(image_dir, num_images=1)
    score_list = []
    for processed_image_path in selected_images:
        original_file_name, original_extension = os.path.splitext(os.path.basename(processed_image_path))
        subdirectory = f'./results/{generation}'
        individual_str = "_".join(individual)
        filename = f'processed_image_{individual_str}_{original_file_name}.png'
        for func in individual:
            if func == 'deblurring':
                processed_image_path = apply_deblurring(processed_image_path)
            elif func == 'denoising':
                processed_image_path = apply_denoising(processed_image_path)
            elif func == 'deraining':
                processed_image_path = apply_deraining(processed_image_path)
            elif func == 'sharpening':
                processed_image_path = apply_sharpening(processed_image_path)
            elif func == 'esrgan':
                processed_image_path = apply_esrgan(processed_image_path)
            elif func == 'diffusion':
                processed_image_path = apply_stable_diffusion(processed_image_path, ddpm_steps=[500])
            elif func == 'empty':
                processed_image_path = processed_image_path
        # Ensure the processed image path exists
        if not os.path.exists(processed_image_path):
            raise FileNotFoundError(f"Processed image not found at {processed_image_path}")
        if not check_image_size(processed_image_path):
            raise ValueError(f"Image size exceeds 20 MB: {processed_image_path}")
        if not is_image_file(processed_image_path):
            processed_image_path = convert_to_supported_format(processed_image_path)

        # Ensure the image is valid by loading it with cv2
        image = cv2.imread(processed_image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {processed_image_path}")
        numbers = list(range(100, 1100, 100))
        ddpm_steps = 400 # random.choice(numbers)
        individual_str = "_".join(individual)
        generation_dir = f'./results/{generation}'

        # Ensure the directory for the current generation exists
        if not os.path.exists(generation_dir):
            os.makedirs(generation_dir)

        # Save the processed image to the appropriate directory with a name based on the individual
        # Define your custom subdirectory and filename


        # Save the processed image
        # processed_image_path = save_processed_image(processed_image_path, subdirectory, filename)
        save_processed_image(processed_image_path, subdirectory, filename)
        # processed_image_path = apply_stable_diffusion(
        #     processed_image_path,
        #     saved_path=f'{generation_dir}/processed_image_{individual_str}.png',
        #     ddpm_steps=[ddpm_steps]
        # )

        url = upload_image_to_cloudinary(processed_image_path)
        scores = get_image_quality_score2(processed_image_path, url)
        score_list.append(np.mean(scores))
    return np.mean(score_list)


def select(population, fitness_scores, num_parents):
    selected_parents = random.choices(population, weights=fitness_scores, k=num_parents)
    return selected_parents
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    # Ensure no adjacent duplicate genes in children
    for i in range(1, len(child1)):
        if child1[i] == child1[i - 1]:
            child1[i] = random.choice([gene for gene in ['deblurring', 'denoising', 'deraining', 'sharpening', 'esrgan', 'diffusion'] if gene != child1[i - 1]])
    for i in range(1, len(child2)):
        if child2[i] == child2[i - 1]:
            child2[i] = random.choice([gene for gene in ['deblurring', 'denoising', 'deraining', 'sharpening', 'esrgan', 'diffusion'] if gene != child2[i - 1]])

    return child1, child2

def mutate(individual, mutation_rate):
    genes = ['deblurring', 'denoising', 'deraining', 'sharpening', 'esrgan', 'diffusion', 'empty']
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            new_gene = random.choice(genes)
            while (new_gene == individual[i] or
                   (i > 0 and new_gene == individual[i - 1]) or
                   (i < len(individual) - 1 and new_gene == individual[i + 1])):
                new_gene = random.choice(genes)
            individual[i] = new_gene
    return individual


def genetic_algorithm(image_dir, pop_size=8, num_generations=200, num_functions=4, mutation_rate=0.1):
    # population = initialize_population(pop_size, num_functions)
    # genes = ['deblurring', 'denoising', 'deraining', 'sharpening']
    # population = initialize_population_with_combinations(pop_size, num_functions, gene)
    population = initialize_population_from_specific_combinations(pop_size, specific_combinations)
    for generation in range(44, num_generations):
        fitness_score_list = [evaluate_fitness(ind, image_dir, generation) for ind in population]
        save_population_and_fitness(population, fitness_score_list, generation)
        new_population = []
        num_parents = pop_size // 2
        parents = select(population, fitness_score_list, num_parents=num_parents)

        if len(parents) % 2 != 0:
            parents.append(parents[0])  # Add the first parent again if odd number of parents

        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        # Ensure the new population size is consistent
        if len(new_population) > pop_size:
            new_population = new_population[:pop_size]
        elif len(new_population) < pop_size:
            while len(new_population) < pop_size:
                new_population.append(mutate(new_population[np.random.randint(len(new_population))], mutation_rate))

        population = new_population
        best_fitness = max(fitness_score_list)
        print(f"Generation {generation}: Best fitness = {best_fitness}")

    # Recalculate fitness for the final population
    final_fitness_scores = [evaluate_fitness(ind, image_dir, num_generations) for ind in population]
    best_individual = population[np.argmax(final_fitness_scores)]

    # Plot results after all generations
    plot_results(num_generations)

    return best_individual


def crop_and_resize_images(input_dir, output_dir, size=512):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        input_image_path = os.path.join(input_dir, filename)

        # Check if the file is an image
        if os.path.isfile(input_image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            output_image_path = os.path.join(output_dir, filename)

            try:
                # Open and process the image
                image = Image.open(input_image_path)
                left = (image.width - image.height) // 2
                top = 0
                right = left + image.height
                bottom = image.height
                cropped_image = image.crop((left, top, right, bottom))
                resized_image = cropped_image.resize((size, size))
                resized_image.save(output_image_path)
                print(f"Cropped and resized image saved to {output_image_path}")
            except Exception as e:
                print(f"Error processing {input_image_path}: {e}")


# Main Workflow
# client_id = "328f6fa73f88463"
crop_and_resize_images("./slected_images", "./inputs")
directory_path = "./inputs"
logging.basicConfig(level=logging.INFO)
# image_urls, image_paths = upload_images_from_directory(directory_path, client_id)

image_urls, image_paths = upload_images_from_directory(directory_path)

best_combination = genetic_algorithm(directory_path)
# num_generations=167
# plot_results(num_generations)
print(f"Best combination: {best_combination}")
