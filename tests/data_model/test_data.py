import email_cleaning_service.data_model.data as data

def test_EmailMessage():
    lines = [
        "This is a test email. I am testing the email cleaner.",
        "This is another line",
        "Yet another line"
    ]
    sections = [4, 5, 6]
    email_message = data.EmailMessage(lines)
    email_message.set_sections(sections)

    assert email_message.get("body") == "This is a test email. I am testing the email cleaner."
    assert email_message.get("signature") == "This is another line"