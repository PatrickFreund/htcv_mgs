import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_pain_distribution_histogram(labels_file, output_folder, pain_threshold=3):
    """
    Create a histogram showing the distribution of images classified as pain vs no pain
    based on the average score across all reviewers.
    
    Args:
        labels_file: Path to the CSV file containing MGS scores
        output_folder: Directory to save the histogram image
        pain_threshold: Threshold score to classify as pain (default: 3)
    """
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(labels_file)
    
    # Find all reviewer column groups
    reviewer_columns = []
    for i in range(1, 20):
        cols = [f'ot{i}', f'nb{i}', f'cb{i}', f'ep{i}', f'wc{i}']
        if all(col in df.columns for col in cols):
            reviewer_columns.append(cols)
    
    print(f"Found {len(reviewer_columns)} reviewers in the data")
    
    # Calculate average scores for each image
    image_avg_scores = {}
    for _, row in df.iterrows():
        image_name = row['index']
        all_reviewer_scores = []
        
        for cols in reviewer_columns:
            try:
                # Get valid scores for this reviewer
                numeric_scores = []
                for s in [row[col] for col in cols]:
                    try:
                        if not pd.isna(s) and s != '-' and s != 9:
                            numeric_scores.append(int(s))
                    except (ValueError, TypeError):
                        pass
                
                if numeric_scores:
                    all_reviewer_scores.append(sum(numeric_scores))
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
        
        if all_reviewer_scores:
            image_avg_scores[image_name] = sum(all_reviewer_scores) / len(all_reviewer_scores)
    
    # Count images in each category
    pain_count = sum(1 for score in image_avg_scores.values() if score >= pain_threshold)
    no_pain_count = sum(1 for score in image_avg_scores.values() if score < pain_threshold)
    
    # Create the histogram
    plt.figure(figsize=(10, 8))
    categories = ['No Pain', 'Pain']
    counts = [no_pain_count, pain_count]
    
    plt.bar(categories, counts, color='#ff5a00ff', alpha=1.0)
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', fontsize=12)
    
    plt.xlabel('Pain Classification', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.title('Distribution of Pain vs No Pain Images', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels
    total_images = pain_count + no_pain_count
    if total_images > 0:
        pain_percent = (pain_count / total_images) * 100
        no_pain_percent = (no_pain_count / total_images) * 100
        plt.annotate(f'{round(no_pain_percent)}%', xy=(0, no_pain_count/2), ha='center', fontsize=12)
        plt.annotate(f'{round(pain_percent)}%', xy=(1, pain_count/2), ha='center', fontsize=12)
    
    # Save the histogram
    output_path = os.path.join(output_folder, f"pain_distribution_{os.path.basename(labels_file).replace('.csv', '')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created pain distribution histogram and saved to {output_path}")
    return {'pain_count': pain_count, 'no_pain_count': no_pain_count}

def create_agreement_percentage_histogram(labels_file, output_folder):
    """
    Create a histogram showing the distribution of exact agreement percentages among reviewers.
    
    Args:
        labels_file: Path to the CSV file containing MGS scores
        output_folder: Directory to save the histogram image
    """
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(labels_file)
    
    # Find all reviewer column groups
    reviewer_columns = []
    for i in range(1, 20):
        cols = [f'ot{i}', f'nb{i}', f'cb{i}', f'ep{i}', f'wc{i}']
        if all(col in df.columns for col in cols):
            reviewer_columns.append(cols)
    
    print(f"Found {len(reviewer_columns)} reviewers in the data")
    
    # Count images by agreement percentage
    agreement_counts = {}
    
    for _, row in df.iterrows():
        pain_count = no_pain_count = total_reviewers = 0
        
        for cols in reviewer_columns:
            try:
                # Get valid scores for this reviewer
                numeric_scores = []
                for s in [row[col] for col in cols]:
                    try:
                        if not pd.isna(s) and s != '-' and s != 9:
                            numeric_scores.append(int(s))
                    except (ValueError, TypeError):
                        pass
                
                if numeric_scores:
                    total_score = sum(numeric_scores)
                    total_reviewers += 1
                    if total_score >= 3:
                        pain_count += 1
                    else:
                        no_pain_count += 1
            except Exception:
                continue
        
        # Calculate agreement percentage for images with multiple reviewers
        if total_reviewers >= 2:
            majority = max(pain_count, no_pain_count)
            agreement_percentage = int((majority / total_reviewers) * 100)
            agreement_counts[agreement_percentage] = agreement_counts.get(agreement_percentage, 0) + 1
    
    # Create histogram
    plt.figure(figsize=(14, 8))
    
    # Sort and prepare data
    sorted_percentages = sorted(agreement_counts.items())
    percentages = [f"{p}%" for p, _ in sorted_percentages]
    counts = [c for _, c in sorted_percentages]
    
    # Create bar chart
    bars = plt.bar(percentages, counts, color='#ff5a00ff')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', fontsize=10)
    
    plt.xlabel('Agreement Percentage Among Reviewers', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.title('Distribution of Reviewer Agreement Percentages', fontsize=16)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the histogram
    output_path = os.path.join(output_folder, f"agreement_percentage_{os.path.basename(labels_file).replace('.csv', '')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created agreement percentage histogram and saved to {output_path}")
    return agreement_counts

def create_experiment_distribution_histogram(main_csv_file, output_folder):
    """
    Create a histogram showing the distribution of images across different experiments.
    
    Args:
        main_csv_file: Path to the main CSV file containing experiment data
        output_folder: Directory to save the histogram image
    """
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(main_csv_file)
    
    # Count images for each experiment
    experiment_counts = df['experiment'].value_counts().sort_index()
    
    # Filter out empty values if any
    experiment_counts = experiment_counts[experiment_counts.index.notnull()]
    
    # Create the histogram
    plt.figure(figsize=(14, 8))
    
    # Create bar chart
    bars = plt.bar(experiment_counts.index, experiment_counts.values, color='#ff5a00ff')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', fontsize=10)
    
    plt.xlabel('Experiment', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.title('Distribution of Images Across Experiments', fontsize=16)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the histogram
    output_path = os.path.join(output_folder, "experiment_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created experiment distribution histogram and saved to {output_path}")
    return dict(experiment_counts)

if __name__ == "__main__":
    # Path to your labels file
    labels_file = 'data/MGS_data/labels/v3_mgs_01.csv'
    main_csv_file = 'data/MGS_data/labels/v3_main.csv'
    
    # Output directory for histograms
    output_folder = 'data/MGS_data/histograms'
    
    # Create pain distribution histogram
    create_pain_distribution_histogram(labels_file, output_folder)
    
    # Create agreement percentage histogram
    create_agreement_percentage_histogram(labels_file, output_folder) 

    # Create experiment distribution histogram
    create_experiment_distribution_histogram(main_csv_file, output_folder)