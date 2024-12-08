# # # from flask import Flask, request, jsonify
# # # from PIL import Image
# # # from transformers import BlipProcessor, BlipForConditionalGeneration
# # # import io
# # # import pickle
# # # import traceback

# # # app = Flask(__name__)

# # # # Load the BLIP model and processor
# # # with open('/Users/shashitejreddysingareddy/Documents/Projects/MY PROJECTS/react learn/react learn/backend/blip_model_processor.pkl', 'rb') as f:
# # #     data = pickle.load(f)
# # #     processor_loaded = data['processor']
# # #     model_loaded = data['model']

# # # @app.route('/Ask', methods=['POST'])
# # # def get_caption():
# # #     try:
# # #         if 'image' not in request.files:
# # #             return jsonify({'error': 'No image file provided'}), 400

# # #         image_file = request.files['image']
# # #         image = Image.open(image_file.stream).convert('RGB')
# # #         inputs = processor_loaded(image, return_tensors="pt")
# # #         out = model_loaded.generate(**inputs)
# # #         # caption = "shashi"
# # #         caption = processor_loaded.decode(out[0], skip_special_tokens=True)
# # #         return jsonify({'caption': caption})
# # #     except Exception as e:
# # #         print("Exception occurred:", e)
# # #         traceback.print_exc()
# # #         return jsonify({'error': str(e)}), 500

# # # if __name__ == '__main__':
# # #     app.run(debug=True,port=8000)



# # # # from flask import Flask
# # # # app = Flask(__name__)
# # # # print("shashi")
# # # # @app.route("/members")
# # # # def members():
# # # #     print("in function")
# # # #     return {"members":["members1","members2"]}

# # # # if __name__ == '__main__':
# # # #     app.run(debug=True,port=8000)






# # # # from flask import Flask, request, jsonify
# # # # from PIL import Image
# # # # from transformers import BlipProcessor, BlipForConditionalGeneration
# # # # import io
# # # # import pickle
# # # # import traceback
# # # # import torch
# # # # from transformers import BlipProcessor
# # # # from transformers import BlipForQuestionAnswering 
# # # # from transformers import BlipImageProcessor 
# # # # from transformers import AutoProcessor
# # # # from transformers import BlipConfig

# # # # app = Flask(__name__)

# # # # # Load the BLIP model and processor
# # # # loaded_model = torch.load("/Users/shashitejreddysingareddy/Documents/Projects/MY PROJECTS/react learn/react learn/backend/BLIP_Model.pkl",map_location=torch.device('cpu'))
# # # # # loaded_model.eval()
# # # # text_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
# # # # image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
# # # # @app.route('/Ask', methods=['POST'])
# # # # def get_caption():
# # # #     try:
# # # #         if 'image' not in request.files:
# # # #             return jsonify({'error': 'No image file provided'}), 400

# # # #         image_file = request.files['image']
# # # #         image = Image.open(image_file.stream).convert('RGB')
# # # #         image_encoding = image_processor(image, do_resize=True, size=(128, 128), return_tensors="pt")
# # # #         question_text = "what is in the image?"
# # # #         encoding = text_processor(
# # # #             None,
# # # #             question_text,
# # # #             padding="max_length",
# # # #             truncation=True,
# # # #             max_length=32,
# # # #             return_tensors="pt"
# # # #         )
# # # #         encoding["pixel_values"] = image_encoding["pixel_values"]
# # # #         with torch.no_grad():
# # # #             outputs = loaded_model.generate(input_ids=encoding['input_ids'], pixel_values=image_encoding['pixel_values'].to('cpu'))

# # # #         predicted_answer = text_processor.decode(outputs[0], skip_special_tokens=True)
# # # #         # return predicted_answer
# # # #         caption = predicted_answer
# # # #         return jsonify({'caption': caption})



# # # #         # inputs = processor_loaded(image, return_tensors="pt")
# # # #         # out = model_loaded.generate(**inputs)
# # # #         # # caption = "shashi"
# # # #         # caption = processor_loaded.decode(out[0], skip_special_tokens=True)
# # # #         # return jsonify({'caption': caption})
# # # #     except Exception as e:
# # # #         print("Exception occurred:", e)
# # # #         traceback.print_exc()
# # # #         return jsonify({'error': str(e)}), 500

# # # # if __name__ == '__main__':
# # # #     app.run(debug=True,port=8000)




# # # from flask import Flask, request, jsonify
# # # from PIL import Image
# # # from transformers import BlipProcessor, BlipForConditionalGeneration
# # # import io
# # # import pickle
# # # import traceback
# # # import torch
# # # import requests
# # # from PIL import Image
# # # app = Flask(__name__)

# # # # Load the BLIP model and processor
# # # loaded_model = torch.load("/Users/shashitejreddysingareddy/Documents/Projects/MY PROJECTS/BLIP_san/blip_model_cap_all.pkl")
# # # loaded_model.eval()

# # # @app.route('/Ask', methods=['POST'])
# # # def get_caption():
# # #     try:
# # #         if 'image' not in request.files:
# # #             return jsonify({'error': 'No image file provided'}), 400
        
# # #         processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# # #         model = loaded_model
# # #         img_path = '/Users/shashitejreddysingareddy/Documents/Projects/MY PROJECTS/BLIP_san/a.png'
# # #         raw_image = Image.open(img_path).convert('RGB')
# # #         text = "a photography of"
# # #         inputs = processor(raw_image, text, return_tensors="pt")

# # #         out = model.generate(**inputs)
# # #         print(processor.decode(out[0], skip_special_tokens=True))

# # #         # unconditional image captioning
# # #         inputs = processor(raw_image, return_tensors="pt")

# # #         out = model.generate(**inputs)
# # #         print(processor.decode(out[0], skip_special_tokens=True))

# # #         return jsonify({'caption': caption})
# # #     except Exception as e:
# # #         print("Exception occurred:", e)
# # #         traceback.print_exc()
# # #         return jsonify({'error': str(e)}), 500

# # # if __name__ == '__main__':
# # #     app.run(debug=True,port=8000)



# # from flask import Flask, request, jsonify
# # from PIL import Image
# # from transformers import BlipProcessor, BlipForConditionalGeneration
# # import torch
# # import io
# # import traceback

# # app = Flask(__name__)

# # # Load the BLIP model and processor
# # def load_model_and_processor():
# #     try:
# #         # Load the BLIP model
# #         model_path = "/Users/shashitejreddysingareddy/Documents/Projects/MY PROJECTS/react learn/react learn/backend/blip_model_cap_all.pkl"
# #         model = torch.load(model_path)
# #         # model.eval()

# #         # Load the BLIP processor
# #         processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        
# #         # Ensure the model is of the correct type
# #         if not isinstance(model, BlipForConditionalGeneration):
# #             raise TypeError("Loaded model is not an instance of BlipForConditionalGeneration")
        
# #         return processor, model
# #     except Exception as e:
# #         print(f"Error loading model and processor: {e}")
# #         traceback.print_exc()
# #         raise

# # processor_loaded, model_loaded = load_model_and_processor()

# # @app.route('/Ask', methods=['POST'])
# # def get_caption():
# #     try:
# #         if 'image' not in request.files:
# #             return jsonify({'error': 'No image file provided'}), 400

# #         image_file = request.files['image']
# #         image = Image.open(image_file.stream).convert('RGB')
        
# #         # Conditional image captioning
# #         text = "a photography of"  # Example conditional text
# #         inputs_conditional = processor_loaded(images=image, text=text, return_tensors="pt")
# #         output_conditional = model_loaded.generate(**inputs_conditional)
# #         caption_conditional = processor_loaded.decode(output_conditional[0], skip_special_tokens=True)
        
# #         # Unconditional image captioning
# #         inputs_unconditional = processor_loaded(images=image, return_tensors="pt")
# #         output_unconditional = model_loaded.generate(**inputs_unconditional)
# #         caption_unconditional = processor_loaded.decode(output_unconditional[0], skip_special_tokens=True)
# #         caption = caption_unconditional
# #         # return jsonify({
# #         #     'conditional_caption': caption_conditional,
# #         #     'unconditional_caption': caption_unconditional
# #         # })
# #         return jsonify({'caption': caption})
# #     except Exception as e:
# #         print("Exception occurred:", e)
# #         traceback.print_exc()
# #         return jsonify({'error': 'Internal Server Error'}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True, port=8000)






print("start")
from flask import Flask, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import io
import traceback
import os
import google.generativeai as genai
# import ollama
import tempfile

app = Flask(__name__)

# Set up Google Generative AI
my_api = "AIzaSyD_jA_sIVHnJPZQDN-2WBligCDG8r6sm7s"
os.environ['GOOGLE_API_KEY'] = my_api
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Load the BLIP model and processor
def load_model_and_processor():
    try:
        # Load the BLIP model
        model_path = "backend/blip_model_cap_all.pkl"
        model = torch.load(model_path)
        # model.eval()

        # Load the BLIP processor
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        
        # Ensure the model is of the correct type
        if not isinstance(model, BlipForConditionalGeneration):
            raise TypeError("Loaded model is not an instance of BlipForConditionalGeneration")
        
        return processor, model
    except Exception as e:
        print(f"Error loading model and processor: {e}")
        traceback.print_exc()
        raise

processor_loaded, model_loaded = load_model_and_processor()

@app.route('/Ask', methods=['POST'])
def get_caption():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')

        temp_image_path = tempfile.mktemp(suffix='.jpg')  # Creates a temporary image file
        image.save(temp_image_path)  # Save image to the file path
        
        # Conditional image captioning using BLIP
        text = "a photography of"  # Example conditional text
        inputs_conditional = processor_loaded(images=image, text=text, return_tensors="pt")
        output_conditional = model_loaded.generate(**inputs_conditional)
        caption_conditional = processor_loaded.decode(output_conditional[0], skip_special_tokens=True)
        print()
        print("caption_conditional : ",caption_conditional)
        print()

        # Unconditional image captioning using BLIP
        inputs_unconditional = processor_loaded(images=image, return_tensors="pt")
        output_unconditional = model_loaded.generate(**inputs_unconditional)
        caption_unconditional = processor_loaded.decode(output_unconditional[0], skip_special_tokens=True)
        print()
        print("caption_unconditional : ",caption_unconditional)
        print()

        # Google Generative AI (Gemini) integration
        vision_model = genai.GenerativeModel('gemini-1.5-flash')
        response = vision_model.generate_content(["Explain the picture?", image])
        google_caption = response.text
        print()
        print("google_caption : ",google_caption)
        print()


        # response = ollama.chat(
        #     model='llama3.2-vision',
        #     messages=[{
        #         'role': 'user',
        #         'content': 'What is in this image?',
        #         'images': temp_image_path
        #     }]
        # )
        # ollama_answwer = response['message']['content']
        # print()
        # print('ollama_answwer : ',ollama_answwer)
        # print()

        # return jsonify({'caption': ollama_answwer})


        
        # Return both BLIP and Google Generative AI captions
        # return jsonify({
        #     'blip_conditional_caption': caption_conditional,
        #     'blip_unconditional_caption': caption_unconditional,
        #     'google_generative_caption': google_caption
        # })
        # return jsonify({'caption': google_caption})
        # return jsonify({'caption': caption_conditional})
        return jsonify({'caption': google_caption})
        
    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)


print("end")





# import os
# import traceback
# import tempfile
# from flask import Flask, request, jsonify
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch
# import google.generativeai as genai
# # import ollama

# app = Flask(__name__)

# # Set up Google Generative AI
# my_api = "AIzaSyD_jA_sIVHnJPZQDN-2WBligCDG8r6sm7s"
# os.environ['GOOGLE_API_KEY'] = my_api
# genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# # Load the BLIP model and processor
# def load_model_and_processor():
#     try:
#         # Load the BLIP model
#         model_path = "/Users/shashitejreddysingareddy/Documents/Projects/MY PROJECTS/react learn/react learn/backend/blip_model_cap_all.pkl"
#         model = torch.load(model_path)
        
#         # Load the BLIP processor
#         processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        
#         # Ensure the model is of the correct type
#         if not isinstance(model, BlipForConditionalGeneration):
#             raise TypeError("Loaded model is not an instance of BlipForConditionalGeneration")
        
#         return processor, model
#     except Exception as e:
#         print(f"Error loading model and processor: {e}")
#         traceback.print_exc()
#         raise

# processor_loaded, model_loaded = load_model_and_processor()

# @app.route('/Ask', methods=['POST'])
# def get_caption():
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image file provided'}), 400

#         image_file = request.files['image']
#         image = Image.open(image_file.stream).convert('RGB')

#         # Define a custom directory where you want to save the image
#         custom_directory = "/Users/shashitejreddysingareddy/Documents/Projects/MY PROJECTS/react learn/react learn/backend/Uploads"
#         if not os.path.exists(custom_directory):
#             os.makedirs(custom_directory)  # Create the directory if it doesn't exist
        
#         # Save the image to a custom path
#         temp_image_path = os.path.join(custom_directory, "temp_image.jpg")  # Path to save the image
#         image.save(temp_image_path)  # Save image to the custom path

#         # Debugging: Check if the image was saved successfully
#         print(f"Temporary image saved at: {temp_image_path}")

#         if not os.path.exists(temp_image_path):
#             return jsonify({'error': 'Temporary image file not found'}), 500

#         # Conditional image captioning using BLIP
#         text = "a photography of"  # Example conditional text
#         inputs_conditional = processor_loaded(images=image, text=text, return_tensors="pt")
#         output_conditional = model_loaded.generate(**inputs_conditional)
#         caption_conditional = processor_loaded.decode(output_conditional[0], skip_special_tokens=True)
#         print("caption_conditional : ", caption_conditional)
#         # return jsonify({'caption': caption_conditional})

#         # Unconditional image captioning using BLIP
#         inputs_unconditional = processor_loaded(images=image, return_tensors="pt")
#         output_unconditional = model_loaded.generate(**inputs_unconditional)
#         caption_unconditional = processor_loaded.decode(output_unconditional[0], skip_special_tokens=True)
#         print("caption_unconditional : ", caption_unconditional)
#         # return jsonify({'caption': caption_unconditional})

#         # # Google Generative AI (Gemini) integration
#         vision_model = genai.GenerativeModel('gemini-1.5-flash')
#         response = vision_model.generate_content(["Explain the picture?", image])
#         google_caption = response.text
#         print("google_caption : ", google_caption)

#         # return jsonify({'caption': google_caption})
#         # Ollama model integration using the image path
#         # response = ollama.chat(
#         #     model='llama3.2-vision',
#         #     messages=[{
#         #         'role': 'user',
#         #         'content': 'What is in this image?',
#         #         'images': ['/Users/shashitejreddysingareddy/Documents/Projects/MY PROJECTS/react learn/react learn/backend/Uploads/temp_image.jpg']
#         #     }]
#         # )
#         # ollama_answer = response['message']['content']
#         # print('ollama_answer : ', ollama_answer)

#         # Clean up the temporary image file (optional)
#         # os.remove(temp_image_path)

#         # return jsonify({'caption': ollama_answer})

#     except Exception as e:
#         print("Exception occurred:", e)
#         traceback.print_exc()
#         return jsonify({'error': 'Internal Server Error'}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=8000)
