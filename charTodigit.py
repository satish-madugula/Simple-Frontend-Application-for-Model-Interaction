import string

# encode the character list into digits list
"""
The below function :
returns the encoded digits of the respective characters

"""
# def encode_char_list(char):


def encodeChar(char):
    char_list = string.ascii_letters
    dig_list = []
    for index, character in enumerate(char):
        try:
            dig_list.append(char_list.index(character))
        except:
            print(character)
    return dig_list
