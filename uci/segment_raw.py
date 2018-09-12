import data
import segmenting


def segment_raw_data():
    """ 
    Segments and compiles raw labelled activity data into a single file
    """
    labelled_activities = data.get_labels()

    segment_arr = segmenting.segment_activities(labelled_activities)

    segmenting.save_segments(segment_arr, 'all_segments.txt')


if __name__ == '__main__':
    # segment_raw_data()
