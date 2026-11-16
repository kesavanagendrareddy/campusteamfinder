# Import Flask and web helper utilities from the flask library
from flask import Flask, render_template, request, jsonify
# Import MongoClient to connect and interact with MongoDB
from pymongo import MongoClient
# Import ObjectId to handle MongoDB document identifier objects
from bson.objectid import ObjectId
# Import TfidfVectorizer to convert text words into numerical values
from sklearn.feature_extraction.text import TfidfVectorizer
# Import cosine_similarity to calculate similarity score between two texts
from sklearn.metrics.pairwise import cosine_similarity
# Import os to access system environment variables if needed
import os

# Initialize the Flask web application
app = Flask(__name__)

# Define the connection URL for the MongoDB server
mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
# Create a client connection to the MongoDB database
client = MongoClient(mongo_uri)
# Access the database named campus_teamfinder
db = client["campus_teamfinder"]
# Access the collection named projects inside the database
projects_collection = db["projects"]

# Helper function to get match score from an applicant for sorting
def get_applicant_score(applicant):
    # Return the match score of the applicant
    return applicant.get("match_score", 0)

# Route 1: Serve the main homepage
@app.route("/")
def index():
    # Render and return the index.html page to the browser
    return render_template("index.html")

# Route 2: Fetch all projects without sensitive creator keys or applicants
@app.route("/api/projects", methods=["GET"])
def get_projects():
    # Fetch all project records from MongoDB
    cursor = projects_collection.find()
    # Create an empty list to store public project data
    projects_list = []
    # Loop through each project found in the database
    for project in cursor:
        # Convert MongoDB ObjectId to a standard string
        project_id = str(project["_id"])
        # Get the list of applicants or an empty list if none exist
        applicants = project.get("applicants", [])
        # Count total applicants for display purposes
        applicant_count = len(applicants)
        # Create a dictionary with only public project information
        public_project = {
            "_id": project_id,
            "title": project.get("title", ""),
            "description": project.get("description", ""),
            "creator_name": project.get("creator_name", ""),
            "required_skills": project.get("required_skills", ""),
            "status": project.get("status", "Open"),
            "applicant_count": applicant_count
        }
        # Add the public project record to our response list
        projects_list.append(public_project)
    # Return the list of projects as a JSON response with status code 200
    return jsonify(projects_list), 200

# Route 3: Create and save a new project to the database
@app.route("/api/projects", methods=["POST"])
def create_project():
    # Extract the incoming JSON data from the request body
    data = request.get_json()
    # Check if request data was received
    if not data:
        # Return an error response if request body is empty
        return jsonify({"error": "No project data received"}), 400
    # Extract the project title from the JSON data
    title = data.get("title", "").strip()
    # Extract the project description from the JSON data
    description = data.get("description", "").strip()
    # Extract the project creator name from the JSON data
    creator_name = data.get("creator_name", "").strip()
    # Extract the security key chosen by the creator
    security_key = data.get("security_key", "").strip()
    # Extract the required skills for this project
    required_skills = data.get("required_skills", "").strip()
    # Verify that the essential fields are provided
    if not title or not description or not security_key:
        # Return an error if required fields are missing
        return jsonify({"error": "Title, description, and security key are required"}), 400
    # Construct the project document according to schema
    new_project = {
        "title": title,
        "description": description,
        "creator_name": creator_name,
        "security_key": security_key,
        "required_skills": required_skills,
        "status": "Open",
        "applicants": []
    }
    # Insert the new project document into the MongoDB collection
    result = projects_collection.insert_one(new_project)
    # Return success response with the new project ID
    return jsonify({"message": "Project created successfully", "project_id": str(result.inserted_id)}), 201

# Route 4: Apply to a specific project by adding an applicant
@app.route("/api/projects/<id>/apply", methods=["POST"])
def apply_to_project(id):
    # Extract the incoming JSON data from the request body
    data = request.get_json()
    # Check if request data was received
    if not data:
        # Return an error response if no data was submitted
        return jsonify({"error": "No applicant data received"}), 400
    # Extract the applicant's full name
    applicant_name = data.get("applicant_name", "").strip()
    # Extract the applicant's skill set
    applicant_skill = data.get("applicant_skill", "").strip()
    # Extract the applicant's pitch explaining why they are a fit
    applicant_pitch = data.get("applicant_pitch", "").strip()
    # Extract the applicant's contact information
    contact = data.get("contact", "").strip()
    # Verify that required applicant fields are filled in
    if not applicant_name or not applicant_pitch:
        # Return an error if name or pitch is missing
        return jsonify({"error": "Applicant name and pitch are required"}), 400
    # Build the applicant dictionary according to schema
    new_applicant = {
        "applicant_name": applicant_name,
        "applicant_skill": applicant_skill,
        "applicant_pitch": applicant_pitch,
        "contact": contact
    }
    # Try converting the URL id parameter to a MongoDB ObjectId
    try:
        # Convert string to ObjectId
        project_object_id = ObjectId(id)
    # Handle error if the id string is not a valid ObjectId
    except Exception:
        # Return an invalid ID format error response
        return jsonify({"error": "Invalid project ID format"}), 400
    # Use PyMongo update_one with $push operator to add applicant to list
    result = projects_collection.update_one(
        {"_id": project_object_id},
        {"$push": {"applicants": new_applicant}}
    )
    # Check if matching project was found in the database
    if result.matched_count == 0:
        # Return error response if no project matched the ID
        return jsonify({"error": "Project not found"}), 404
    # Return a success message response with status code 200
    return jsonify({"message": "Application submitted successfully"}), 200

# Route 5: View applicants with boosted ML match scores after checking security key
@app.route("/api/projects/<id>/view_applicants", methods=["POST"])
def view_applicants(id):
    # Extract the incoming JSON data containing the security key
    data = request.get_json()
    # Check if JSON payload exists
    if not data:
        # Return error if request body is empty
        return jsonify({"error": "Security key is required"}), 400
    # Extract the security key submitted by user
    entered_key = data.get("security_key", "").strip()
    # Try converting the URL id string to a MongoDB ObjectId
    try:
        # Convert string to ObjectId
        project_object_id = ObjectId(id)
    # Handle error if the project ID string format is invalid
    except Exception:
        # Return an invalid ID format error
        return jsonify({"error": "Invalid project ID format"}), 400
    # Query MongoDB for the project with this ObjectId
    project = projects_collection.find_one({"_id": project_object_id})
    # Check if the project exists in the database
    if not project:
        # Return 404 error if project was not found
        return jsonify({"error": "Project not found"}), 404
    # Check if the entered security key matches the project's saved security key
    if project.get("security_key") != entered_key:
        # Return unauthorized 403 error if keys do not match
        return jsonify({"error": "Invalid security key. Access denied."}), 403
    # Retrieve the applicants list from the project document
    applicants = project.get("applicants", [])
    # Check if there are no applicants yet
    if len(applicants) == 0:
        # Return an empty list response if no one applied yet
        return jsonify([]), 200
    # Try running the boosted ML match algorithm
    try:
        # Start ML code: Create a list of texts starting with project description
        text_corpus = [project.get("description", "")]
        # Loop through each applicant to collect their pitch text
        for applicant in applicants:
            # Add applicant's pitch to the text corpus list
            text_corpus.append(applicant.get("applicant_pitch", ""))
        # Initialize TF-IDF Vectorizer with English stop words filter
        vectorizer = TfidfVectorizer(stop_words="english")
        # Fit vectorizer and transform text into numerical TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(text_corpus)
        # Compute cosine similarity between project description (row 0) and all pitches (row 1+)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        # Loop through each applicant to attach their boosted match score
        for i in range(len(applicants)):
            # Calculate boosted score with 2.5x multiplier and cap at 100 percent
            applicants[i]["match_score"] = min(int(float(scores[i]) * 100 * 2.5), 100)
    # Catch any potential ML processing error and set default score
    except Exception:
        # Loop through applicants to set fallback score
        for applicant in applicants:
            # Set default match score to zero
            applicant["match_score"] = 0
    # Sort the applicants list in place from highest match score to lowest
    applicants.sort(key=get_applicant_score, reverse=True)
    # Return the ranked list of applicants with their match scores as JSON
    return jsonify(applicants), 200

# Route 6: Delete a project after verifying security key
@app.route("/api/projects/<id>", methods=["DELETE"])
def delete_project(id):
    # Extract the incoming JSON data containing the security key
    data = request.get_json()
    # Check if request data exists
    if not data:
        # Return error if request body is empty
        return jsonify({"error": "Security key is required"}), 400
    # Extract the security key submitted by the user
    entered_key = data.get("security_key", "").strip()
    # Try converting the URL id parameter to a MongoDB ObjectId
    try:
        # Convert string to ObjectId
        project_object_id = ObjectId(id)
    # Handle error if the project ID string format is invalid
    except Exception:
        # Return an invalid ID format error
        return jsonify({"error": "Invalid project ID format"}), 400
    # Query MongoDB for the project with this ObjectId
    project = projects_collection.find_one({"_id": project_object_id})
    # Check if the project exists in the database
    if not project:
        # Return 404 error if project was not found
        return jsonify({"error": "Project not found"}), 404
    # Check if the entered security key matches the project's saved security key
    if project.get("security_key") != entered_key:
        # Return unauthorized 403 error if keys do not match
        return jsonify({"error": "Invalid security key. Cannot delete project."}), 403
    # Delete the project document from MongoDB using delete_one
    projects_collection.delete_one({"_id": project_object_id})
    # Return success response confirming project deletion
    return jsonify({"message": "Project deleted successfully"}), 200

# Check if script is run directly from the command line
if __name__ == "__main__":
    # Start the Flask web development server on port 5000 in debug mode
    app.run(host="0.0.0.0", port=5000, debug=True)
