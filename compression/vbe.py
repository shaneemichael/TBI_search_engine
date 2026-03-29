import array

class VBEPostings:
    """ 
    Unlike StandardPostings, where for a postings list,
    what is stored on disk is the original sequence of integers from the postings
    list as is.

    In VBEPostings, this time, what is stored is its gap, except
    the first posting. Only after that is it encoded with the Variable-Byte
    Encoding algorithm to a bytestream.

    Example:
    postings list [34, 67, 89, 454] will first be converted to gap-based,
    which is [34, 33, 22, 365]. Only then is it encoded with the
    Variable-Byte Encoding compression algorithm, and then converted to a bytestream.

    ASSUMPTION: the postings_list for a term FITS in memory!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        See our textbook!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend to the front
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # the first bit of the last byte is changed to 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Performs encoding (of course with compression) on a
        list of numbers, with Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a stream of bytes (with Variable-Byte
        Encoding). DO NOT FORGET to convert it first to a gap-based list, before
        encoding it and converting it to a bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray representing the sequence of integers in postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies into a stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray representing the raw TF value of term occurrences in each
            document in the postings list
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding a bytestream that was previously encoded with
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list from a stream of bytes. DO NOT FORGET
        the bytestream decoded from encoded_postings_list is still a
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray representing the encoded postings list as output
            from the encode static method above.

        Returns
        -------
        List[int]
            list of docIDs that is the result of decoding encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies from a stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray representing the encoded term frequencies list as output
            from the encode_tf static method above.

        Returns
        -------
        List[int]
            List of term frequencies that is the result of decoding encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)
