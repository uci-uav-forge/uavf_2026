from perception.camera.zr10.utils import crc16_str_swap


def test_crc16():
    data = "5566010100000005FF"
    expected_crc = "5c6a"
    data_crc = crc16_str_swap(data)

    assert data_crc == expected_crc, (
        "CRC16 of data %s did not match expected CRC %s" % (data_crc, expected_crc)
    )
